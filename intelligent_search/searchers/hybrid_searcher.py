#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合检索器
结合关键词检索和向量检索，实现混合检索功能
"""

import sys
import os
from typing import List, Dict, Any, Optional, Union

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from .base_searcher import BaseSearcher, SearchResult
from .keyword_searcher import KeywordSearcher
from .vector_searcher import VectorSearcher


class HybridSearcher(BaseSearcher):
    """混合检索器，结合关键词和向量检索"""
    
    def __init__(self):
        """初始化混合检索器"""
        # 初始化两个子检索器
        self.keyword_searcher = KeywordSearcher()
        self.vector_searcher = VectorSearcher()
        
        print("[HybridSearcher] 混合检索器初始化完成")
    
    def _initialize_connection(self):
        """初始化连接（由子检索器处理）"""
        pass
    
    def search(
        self,
        query: Union[str, List[str]],
        fund_code: Optional[str] = None,
        source_file: Optional[str] = None,
        top_k: int = 10,
        chunk_range: Optional[tuple] = None,
        intent: str = "content",
        **kwargs
    ) -> List[SearchResult]:
        """
        执行混合检索
        
        Args:
            query: 查询字符串或关键词列表
            fund_code: 基金代码过滤
            source_file: 源文件过滤
            top_k: 每种检索器的候选数量上限，合并后可能多于该值
            
        Returns:
            List[SearchResult]: 混合检索结果列表
        """
        
        print(f"[HybridSearcher] 开始混合检索: {query}, 意图={intent}")
        
        try:
            # 处理查询字符串
            if isinstance(query, list):
                query_text = " ".join(str(item) for item in query)
            else:
                query_text = str(query) if query is not None else ""

            query_text = query_text.strip()

            if intent == "title":
                keyword_query = query_text
            else:
                if isinstance(query, list):
                    keyword_query = [str(item) for item in query]
                else:
                    keyword_query = [token for token in query_text.split() if token]

            # 1. 执行向量检索
            print("[HybridSearcher] 执行向量检索...")
            vector_results = self.vector_searcher.search(
                query=query_text,
                fund_code=fund_code,
                source_file=source_file,
                top_k=top_k,
                chunk_range=chunk_range,
                intent=intent
            )
            
            # 2. 执行关键词检索
            print("[HybridSearcher] 执行关键词检索...")
            keyword_results = self.keyword_searcher.search(
                query=keyword_query,
                fund_code=fund_code,
                source_file=source_file,
                top_k=top_k,
                chunk_range=chunk_range,
                intent=intent
            )
            
            # 3. 合并去重（最多保留两路检索的全部不重复结果）
            merged_results = self._merge_and_deduplicate(vector_results, keyword_results)

            final_results = merged_results
            print(f"[HybridSearcher] 混合检索完成，返回 {len(final_results)} 条结果")
            return final_results
            
        except Exception as e:
            print(f"[HybridSearcher] 混合检索失败: {e}")
            return []
    
    def _merge_and_deduplicate(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult]
    ) -> List[SearchResult]:
        """合并两种检索结果并去重"""
        
        print(f"[HybridSearcher] 开始合并去重...")
        print(f"  - 向量检索结果: {len(vector_results)} 条")
        print(f"  - 关键词检索结果: {len(keyword_results)} 条")
        
        all_results = []
        seen_ids = set()
        duplicate_count = 0
        
        # 统计各种来源的结果数量
        vector_only = []
        keyword_only = []
        both_methods = []
        
        # 1. 先添加向量检索结果（优先级高）
        for result in vector_results:
            if result.global_id not in seen_ids:
                result.from_methods = ["vector"]
                all_results.append(result)
                seen_ids.add(result.global_id)
                vector_only.append(result)
            else:
                duplicate_count += 1
        
        # 2. 再添加关键词检索结果
        for result in keyword_results:
            if result.global_id not in seen_ids:
                result.from_methods = ["keyword"]
                all_results.append(result)
                seen_ids.add(result.global_id)
                keyword_only.append(result)
            else:
                # 标记为两种方法都命中的结果
                duplicate_count += 1
                for existing in all_results:
                    if existing.global_id == result.global_id:
                        if "keyword" not in existing.from_methods:
                            existing.from_methods.append("keyword")
                            both_methods.append(existing)
                        break
        
        # 输出详细统计信息
        print(f"[HybridSearcher] 去重统计:")
        print(f"  - 去重前总数: {len(vector_results) + len(keyword_results)} 条")
        print(f"  - 去重后总数: {len(all_results)} 条")
        print(f"  - 重复项数量: {duplicate_count} 条")
        
        print(f"[HybridSearcher] 结果来源分布:")
        print(f"  - 仅向量检索: {len(vector_only)} 条")
        print(f"  - 仅关键词检索: {len(keyword_only)} 条")
        print(f"  - 两种方法都命中: {len(both_methods)} 条")
        
        # 按分数排序（向量检索结果在前，相同来源内按分数降序）
        all_results.sort(key=lambda x: (
            0 if "vector" in x.from_methods else 1,  # 向量结果优先
            -x.score  # 分数降序
        ))
        
        return all_results
    
    def get_search_statistics(
        self,
        query: Union[str, List[str]],
        fund_code: Optional[str] = None,
        source_file: Optional[str] = None,
        top_k: int = 10,
        intent: str = "content"
    ) -> Dict[str, Any]:
        """获取检索统计信息（用于分析和调试）"""
        
        # 处理查询字符串
        if isinstance(query, list):
            query_text = " ".join(str(item) for item in query)
        else:
            query_text = str(query) if query is not None else ""

        query_text = query_text.strip()

        if intent == "title":
            keyword_query = query_text
        else:
            if isinstance(query, list):
                keyword_query = [str(item) for item in query]
            else:
                keyword_query = [token for token in query_text.split() if token]
        
        # 分别执行两种检索
        vector_results = self.vector_searcher.search(
            query=query_text,
            fund_code=fund_code,
            source_file=source_file,
            top_k=top_k,
            intent=intent
        )
        
        keyword_results = self.keyword_searcher.search(
            query=keyword_query,
            fund_code=fund_code,
            source_file=source_file,
            top_k=top_k,
            intent=intent
        )
        
        # 分析重叠情况
        vector_ids = {r.global_id for r in vector_results}
        keyword_ids = {r.global_id for r in keyword_results}
        overlap_ids = vector_ids & keyword_ids
        
        return {
            "vector_count": len(vector_results),
            "keyword_count": len(keyword_results),
            "overlap_count": len(overlap_ids),
            "overlap_ratio": len(overlap_ids) / max(len(vector_ids | keyword_ids), 1),
            "vector_only": len(vector_ids - keyword_ids),
            "keyword_only": len(keyword_ids - vector_ids),
            "vector_top3_scores": [r.score for r in vector_results[:3]],
            "keyword_top3_scores": [r.score for r in keyword_results[:3]]
        }
    
    def close_connection(self):
        """关闭所有连接"""
        try:
            self.keyword_searcher.close_connection()
            self.vector_searcher.close_connection()
            print("[HybridSearcher] 所有连接已关闭")
        except Exception as e:
            print(f"[HybridSearcher] 关闭连接时出错: {e}")