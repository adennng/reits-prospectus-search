#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量检索器
基于Milvus向量数据库实现语义检索功能
"""

import sys
import os
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection
from openai import OpenAI

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from db_config import get_vector_db_config
from model_config import MODEL_CONFIG
from .base_searcher import BaseSearcher, SearchResult


class VectorSearcher(BaseSearcher):
    """向量检索器，基于Milvus向量数据库"""
    
    def __init__(self):
        """初始化向量检索器"""
        self.vector_config = get_vector_db_config()
        
        # 初始化embedding模型客户端
        self._init_embedding_client()
        
        # Milvus配置 - 在调用super().__init__之前设置
        self.collection_name = "reits_announcement"  
        self.alias_name = "intelligent_search_connection"
        
        # 调用父类初始化（会调用_initialize_connection）
        super().__init__(self.vector_config)
        
        print("[VectorSearcher] 向量检索器初始化完成")
    
    def _initialize_connection(self):
        """初始化Milvus连接"""
        try:
            connections.connect(
                alias=self.alias_name,
                host=self.vector_config['host'],
                port=self.vector_config['port'],
                user=self.vector_config['user'],
                password=self.vector_config['password']
            )
            
            # 初始化Collection
            self._connection = Collection(
                name=self.collection_name,
                using=self.alias_name
            )
            
            print("[VectorSearcher] Milvus连接成功")
            
        except Exception as e:
            print(f"[VectorSearcher] Milvus连接失败: {e}")
            raise e
    
    def _init_embedding_client(self):
        """初始化embedding模型客户端"""
        try:
            # 使用智谱AI的embedding模型
            embedding_config = MODEL_CONFIG["zhipu"]["embedding-3"]
            self.embedding_client = OpenAI(
                api_key=embedding_config["api_key"],
                base_url=embedding_config["base_url"]
            )
            self.embedding_model = embedding_config["model"]
            print("[VectorSearcher] Embedding客户端初始化成功")
            
        except Exception as e:
            print(f"[VectorSearcher] Embedding客户端初始化失败: {e}")
            raise e
    
    def search(
        self,
        query: str,
        fund_code: Optional[str] = None,
        source_file: Optional[str] = None,
        top_k: int = 10,
        chunk_range: Optional[tuple] = None,
        intent: str = "content",
        **kwargs
    ) -> List[SearchResult]:
        """
        执行向量检索
        
        Args:
            query: 查询字符串
            fund_code: 基金代码过滤
            source_file: 源文件过滤
            top_k: 返回结果数量
            
        Returns:
            List[SearchResult]: 检索结果列表
        """
        
        query = "" if query is None else str(query).strip()
        print(f"[VectorSearcher] 开始向量检索: {query[:50]}..., 意图={intent}")
        
        try:
            # 1. 生成查询向量
            query_vector = self._generate_embedding(query)
            if not query_vector:
                print("[VectorSearcher] 查询向量生成失败")
                return []
            
            # 2. 构建搜索参数
            search_params = self._build_search_params()
            
            # 3. 构建过滤表达式
            expr = self._build_filter_expression(
                fund_code=fund_code,
                source_file=source_file,
                chunk_range=chunk_range,
                intent=intent
            )
            
            # 4. 执行向量搜索
            results = self._connection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=[
                    "global_id", "chunk_id", "source_file", "page_num",
                    "text", "fund_code", "date", "short_name"
                ]
            )
            
            # 5. 处理搜索结果
            formatted_results = self._process_search_results(results)
            
            print(f"[VectorSearcher] 向量检索完成，返回 {len(formatted_results)} 条结果")
            return formatted_results
            
        except Exception as e:
            print(f"[VectorSearcher] 向量检索失败: {e}")
            return []
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """生成文本的向量表示"""
        
        try:
            # 限制文本长度，避免超出模型限制
            if len(text) > 8000:
                text = text[:8000]
            
            response = self.embedding_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            
            embedding = response.data[0].embedding
            print(f"[VectorSearcher] 向量生成成功，维度: {len(embedding)}")
            return embedding
            
        except Exception as e:
            print(f"[VectorSearcher] 向量生成失败: {e}")
            return None
    
    def _build_search_params(self) -> Dict[str, Any]:
        """构建向量搜索参数"""
        return {
            "metric_type": "L2",  # 使用L2距离
            "params": {
                "nprobe": 16  # 平衡检索速度和精度
            }
        }
    
    def _build_filter_expression(
        self,
        fund_code: Optional[str] = None,
        source_file: Optional[str] = None,
        chunk_range: Optional[tuple] = None,
        intent: str = "content"
    ) -> Optional[str]:
        """构建Milvus过滤表达式"""

        conditions = []

        _ = intent  # 保留参数，便于根据意图调整过滤策略

        if fund_code:
            conditions.append(f'fund_code == "{fund_code}"')

        if source_file:
            conditions.append(f'source_file == "{source_file}"')

        if chunk_range and any(value is not None for value in chunk_range):
            start_chunk, end_chunk = chunk_range
            if start_chunk is not None:
                conditions.append(f'chunk_id >= {start_chunk}')
            if end_chunk is not None:
                conditions.append(f'chunk_id <= {end_chunk}')

        if conditions:
            return " and ".join(conditions)
        else:
            return None

    def _process_search_results(self, milvus_results) -> List[SearchResult]:
        """处理Milvus检索结果"""
        
        results = []
        
        for hits in milvus_results:
            for hit in hits:
                # 转换L2距离为相似度分数
                distance = hit.distance
                score = 1.0 / (1.0 + distance)  # 距离越小，分数越高
                
                # 构建结果数据
                result_data = {
                    '_source': {
                        'global_id': hit.entity.get('global_id', ''),
                        'chunk_id': hit.entity.get('chunk_id', 0),
                        'source_file': hit.entity.get('source_file', ''),
                        'page_num': hit.entity.get('page_num', ''),
                        'text': hit.entity.get('text', ''),
                        'fund_code': hit.entity.get('fund_code', ''),
                        'date': hit.entity.get('date', ''),
                        'short_name': hit.entity.get('short_name', '')
                    }
                }
                
                search_result = self._format_search_result(result_data, score, 'vector')
                results.append(search_result)
        
        return results
    
    def close_connection(self):
        """关闭Milvus连接"""
        try:
            if hasattr(connections, 'disconnect'):
                connections.disconnect(alias=self.alias_name)
                print("[VectorSearcher] Milvus连接已关闭")
        except Exception as e:
            print(f"[VectorSearcher] 关闭连接时出错: {e}")
