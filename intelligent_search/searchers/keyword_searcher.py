#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
关键词检索器
基于Elasticsearch实现关键词检索功能
"""

import sys
import os
from typing import List, Dict, Any, Optional, Union
from elasticsearch import Elasticsearch

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from db_config import get_elasticsearch_config
from .base_searcher import BaseSearcher, SearchResult


class KeywordSearcher(BaseSearcher):
    """关键词检索器，基于Elasticsearch"""
    
    def __init__(self):
        """初始化关键词检索器"""
        self.es_config = get_elasticsearch_config()
        super().__init__(self.es_config)
        self.index_name = "reits_announcements"
        print("[KeywordSearcher] 关键词检索器初始化完成")
    
    def _initialize_connection(self):
        """初始化ES连接"""
        try:
            self._connection = Elasticsearch(
                [f"{self.es_config['scheme']}://{self.es_config['host']}:{self.es_config['port']}"],
                basic_auth=(self.es_config['username'], self.es_config['password']),
                verify_certs=False,
                ssl_show_warn=False
            )
            print("[KeywordSearcher] Elasticsearch连接成功")
        except Exception as e:
            print(f"[KeywordSearcher] Elasticsearch连接失败: {e}")
            raise e
    
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
        执行关键词检索
        
        Args:
            query: 查询字符串或关键词列表
            fund_code: 基金代码过滤
            source_file: 源文件过滤
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 检索结果列表
        """
        
        print(f"[KeywordSearcher] 开始关键词检索: {query}, 意图={intent}")
        
        try:
            # 构建查询
            search_body = self._build_search_query(
                query, fund_code, source_file, top_k, chunk_range, intent
            )
            
            # 执行搜索
            response = self._connection.search(
                index=self.index_name,
                body=search_body
            )
            
            # 处理结果
            results = self._process_search_results(response)
            
            print(f"[KeywordSearcher] 关键词检索完成，返回 {len(results)} 条结果")
            return results
            
        except Exception as e:
            print(f"[KeywordSearcher] 关键词检索失败: {e}")
            return []
    
    def _build_search_query(
        self,
        query: Union[str, List[str]],
        fund_code: Optional[str] = None,
        source_file: Optional[str] = None,
        top_k: int = 10,
        chunk_range: Optional[tuple] = None,
        intent: str = "content"
    ) -> Dict[str, Any]:
        """构建ES查询体"""
        
        # 处理查询字符串
        if isinstance(query, list):
            query_text = " ".join(str(item) for item in query)
        else:
            query_text = str(query) if query is not None else ""

        query_text = query_text.strip()

        if intent == "title":
            base_query = {
                "match_phrase": {
                    "text": query_text
                }
            }
        else:
            base_query = {
                "multi_match": {
                    "query": query_text,
                    "fields": ["text"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            }
        
        print(f"[KeywordSearcher] 使用intent={intent}, 查询文本预览: {query_text[:80]}")

        # 构建过滤条件
        must_filters = []
        
        if fund_code:
            must_filters.append({"term": {"fund_code": fund_code}})
        
        if source_file:
            # 尝试精确匹配和keyword字段匹配
            source_file_filter = {
                "bool": {
                    "should": [
                        {"term": {"source_file.keyword": source_file}},
                        {"term": {"source_file": source_file}}
                    ]
                }
            }
            must_filters.append(source_file_filter)

        if chunk_range:
            chunk_filter = {"range": {"chunk_id": {}}}
            start_chunk, end_chunk = chunk_range
            if start_chunk is not None:
                chunk_filter['range']['chunk_id']['gte'] = start_chunk
            if end_chunk is not None:
                chunk_filter['range']['chunk_id']['lte'] = end_chunk
            if chunk_filter['range']['chunk_id']:
                must_filters.append(chunk_filter)
        
        # 组装最终查询
        if must_filters:
            search_query = {
                "bool": {
                    "must": [base_query],
                    "filter": must_filters
                }
            }
        else:
            search_query = base_query
        
        # 完整查询体
        search_body = {
            "size": top_k,
            "query": search_query,
            "_source": [
                "global_id", "chunk_id", "source_file", "page_num", 
                "text", "fund_code", "date", "short_name"
            ],
            "sort": [
                {"_score": {"order": "desc"}}
            ]
        }
        
        return search_body
    
    def _process_search_results(self, response: Dict) -> List[SearchResult]:
        """处理ES检索结果"""
        
        results = []
        hits = response.get('hits', {}).get('hits', [])
        
        for hit in hits:
            score = hit.get('_score', 0.0)
            search_result = self._format_search_result(hit, score, 'keyword')
            results.append(search_result)
        
        return results
    
    def get_file_chunks(
        self,
        fund_code: str,
        source_file: str,
        sort_by_chunk_id: bool = True
    ) -> List[SearchResult]:
        """获取指定文件的所有语块"""
        
        print(f"[KeywordSearcher] 获取文件语块: {source_file}")
        
        try:
            search_body = {
                "size": 10000,  # 获取所有语块
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"fund_code": fund_code}},
                            {
                                "bool": {
                                    "should": [
                                        {"term": {"source_file.keyword": source_file}},
                                        {"term": {"source_file": source_file}}
                                    ]
                                }
                            }
                        ]
                    }
                },
                "_source": [
                    "global_id", "chunk_id", "source_file", "page_num", 
                    "text", "fund_code", "date", "short_name"
                ]
            }
            
            if sort_by_chunk_id:
                search_body["sort"] = [{"chunk_id": {"order": "asc"}}]
            
            response = self._connection.search(
                index=self.index_name,
                body=search_body
            )
            
            results = []
            hits = response.get('hits', {}).get('hits', [])
            
            for hit in hits:
                search_result = self._format_search_result(hit, 1.0, 'keyword')
                results.append(search_result)
            
            print(f"[KeywordSearcher] 获取到 {len(results)} 个语块")
            return results
            
        except Exception as e:
            print(f"[KeywordSearcher] 获取文件语块失败: {e}")
            return []