#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语块处理工具模块
包含语块过滤、扩展、合并等功能
"""

from typing import List, Optional, Dict, Any
from .page_utils import PageUtils


class ChunkUtils:
    """语块处理工具类"""
    
    @staticmethod
    def apply_range_limitations(
        chunks: List,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        start_chunk_id: Optional[int] = None,
        end_chunk_id: Optional[int] = None
    ) -> List:
        """应用页码和chunk_id范围限制"""
        
        if not chunks:
            return chunks
        
        print(f"[ChunkUtils] 应用范围限制: 页码[{start_page}-{end_page}], chunk_id[{start_chunk_id}-{end_chunk_id}]")
        
        filtered_chunks = chunks
        
        # 应用chunk_id范围限制
        if start_chunk_id is not None or end_chunk_id is not None:
            chunk_filtered = []
            for chunk in filtered_chunks:
                # 兼容SearchResult对象和字典格式
                if hasattr(chunk, 'chunk_id'):
                    chunk_id = chunk.chunk_id
                else:
                    chunk_id = chunk['_source']['chunk_id']
                    
                if start_chunk_id is not None and chunk_id < start_chunk_id:
                    continue
                if end_chunk_id is not None and chunk_id > end_chunk_id:
                    continue
                chunk_filtered.append(chunk)
            filtered_chunks = chunk_filtered
            print(f"[ChunkUtils] chunk_id范围过滤后保留 {len(filtered_chunks)} 个语块")
        
        # 应用页码范围限制
        if start_page is not None or end_page is not None:
            page_filtered = []
            for chunk in filtered_chunks:
                # 兼容SearchResult对象和字典格式
                if hasattr(chunk, 'page_num'):
                    page_num_str = chunk.page_num
                else:
                    page_num_str = chunk['_source'].get('page_num', '')
                    
                chunk_pages = PageUtils.extract_page_numbers_from_string(page_num_str)
                
                # 检查该语块是否与指定页码范围有交集
                if chunk_pages:
                    min_page = min(chunk_pages)
                    max_page = max(chunk_pages)
                    
                    # 判断是否在范围内
                    if start_page is not None and max_page < start_page:
                        continue
                    if end_page is not None and min_page > end_page:
                        continue
                    
                page_filtered.append(chunk)
            filtered_chunks = page_filtered
            print(f"[ChunkUtils] 页码范围过滤后保留 {len(filtered_chunks)} 个语块")
        
        return filtered_chunks
    
    @staticmethod
    def expand_chunks(
        target_chunks: List,
        all_chunks: List, 
        expand_before: int = 0,
        expand_after: int = 0
    ) -> List:
        """扩展目标语块，向前向后获取更多上下文"""
        
        if not target_chunks or (expand_before == 0 and expand_after == 0):
            return target_chunks
        
        print(f"[ChunkUtils] 扩展语块: 向前{expand_before}块, 向后{expand_after}块")
        
        # 获取目标语块的chunk_id范围
        target_chunk_ids = set()
        for chunk in target_chunks:
            if hasattr(chunk, 'chunk_id'):
                target_chunk_ids.add(chunk.chunk_id)
            else:
                target_chunk_ids.add(chunk['_source']['chunk_id'])
        
        min_chunk_id = min(target_chunk_ids)
        max_chunk_id = max(target_chunk_ids)
        
        # 计算扩展后的范围
        expand_start_id = min_chunk_id - expand_before
        expand_end_id = max_chunk_id + expand_after
        
        # 从全部语块中筛选扩展范围内的语块
        expanded_chunks = []
        for chunk in all_chunks:
            if hasattr(chunk, 'chunk_id'):
                chunk_id = chunk.chunk_id
            else:
                chunk_id = chunk['_source']['chunk_id']
            
            if expand_start_id <= chunk_id <= expand_end_id:
                expanded_chunks.append(chunk)
        
        # 按chunk_id排序
        expanded_chunks.sort(key=lambda x: x.chunk_id if hasattr(x, 'chunk_id') else x['_source']['chunk_id'])
        
        print(f"[ChunkUtils] 扩展后获得 {len(expanded_chunks)} 个语块")
        return expanded_chunks
    
    @staticmethod
    def merge_chunks_text(chunks: List) -> str:
        """合并多个语块的文本内容"""
        
        if not chunks:
            return ""
        
        # 按chunk_id排序确保顺序正确
        sorted_chunks = sorted(chunks, key=lambda x: x.chunk_id if hasattr(x, 'chunk_id') else x['_source']['chunk_id'])
        
        # 拼接文本
        texts = []
        for chunk in sorted_chunks:
            if hasattr(chunk, 'text'):
                texts.append(chunk.text)
            else:
                texts.append(chunk['_source']['text'])
        
        merged_text = "".join(texts)
        return merged_text
    
    @staticmethod
    def get_chunk_id_range_from_chunks(chunks: List) -> tuple:
        """计算语块列表的chunk_id范围，返回(最小chunk_id, 最大chunk_id)"""
        
        if not chunks:
            return None, None
        
        chunk_ids = []
        for chunk in chunks:
            if hasattr(chunk, 'chunk_id'):
                chunk_ids.append(chunk.chunk_id)
            else:
                chunk_ids.append(chunk['_source']['chunk_id'])
        
        return min(chunk_ids), max(chunk_ids)
    
    @staticmethod
    def filter_chunks_by_page_range(chunks: List, start_page: int, end_page: int) -> List:
        """根据页码范围过滤语块"""
        
        filtered_chunks = []
        for chunk in chunks:
            # 兼容SearchResult对象和字典格式
            if hasattr(chunk, 'page_num'):
                page_num_str = chunk.page_num
            else:
                page_num_str = chunk['_source'].get('page_num', '')
                
            chunk_pages = PageUtils.extract_page_numbers_from_string(page_num_str)
            
            if chunk_pages:
                min_page = min(chunk_pages)
                max_page = max(chunk_pages)
                
                # 判断是否在范围内（有交集即保留）
                if max_page >= start_page and min_page <= end_page:
                    filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    @staticmethod
    def filter_chunks_by_chunk_id_range(chunks: List, start_chunk_id: int, end_chunk_id: int) -> List:
        """根据chunk_id范围过滤语块"""
        
        filtered_chunks = []
        for chunk in chunks:
            # 兼容SearchResult对象和字典格式
            if hasattr(chunk, 'chunk_id'):
                chunk_id = chunk.chunk_id
            else:
                chunk_id = chunk['_source']['chunk_id']
                
            if start_chunk_id <= chunk_id <= end_chunk_id:
                filtered_chunks.append(chunk)
        
        return filtered_chunks