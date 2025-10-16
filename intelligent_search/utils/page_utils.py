#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
页码处理工具模块
包含页码提取、范围计算、页码查找等功能
"""

from typing import List, Optional, Dict, Any
import re


class PageUtils:
    """页码处理工具类"""
    
    @staticmethod
    def extract_page_numbers_from_string(page_num_str: str) -> List[int]:
        """从page_num字符串中提取所有页码数字"""
        
        if not page_num_str:
            return []
        
        pages = []
        # 分割"-"获取各个页码
        for part in str(page_num_str).split('-'):
            try:
                page_num = int(part.strip())
                pages.append(page_num)
            except:
                continue
        
        return pages
    
    @staticmethod
    def calculate_page_range(chunks: List) -> tuple:
        """计算语块列表的页码范围"""
        
        all_pages = []
        for chunk in chunks:
            # 兼容SearchResult对象和字典格式
            if hasattr(chunk, 'page_num'):
                page_num_str = chunk.page_num
            else:
                page_num_str = chunk['_source'].get('page_num', '')
            pages = PageUtils.extract_page_numbers_from_string(page_num_str)
            all_pages.extend(pages)
        
        if all_pages:
            return min(all_pages), max(all_pages)
        else:
            return None, None
    
    @staticmethod
    def find_first_chunk_containing_page(all_chunks: List, target_page: int) -> Optional[Dict]:
        """找到page_num中第一个包含目标页码的语块"""
        
        for chunk in all_chunks:
            # 兼容SearchResult对象和字典格式
            if hasattr(chunk, 'page_num'):
                page_num_str = chunk.page_num
            else:
                page_num_str = chunk['_source'].get('page_num', '')
            chunk_pages = PageUtils.extract_page_numbers_from_string(page_num_str)
            
            if target_page in chunk_pages:
                return chunk
        
        return None
    
    @staticmethod
    def find_last_chunk_containing_page(all_chunks: List, target_page: int) -> Optional[Dict]:
        """找到page_num中最后一个包含目标页码的语块"""
        
        last_chunk = None
        for chunk in all_chunks:
            # 兼容SearchResult对象和字典格式
            if hasattr(chunk, 'page_num'):
                page_num_str = chunk.page_num
            else:
                page_num_str = chunk['_source'].get('page_num', '')
            chunk_pages = PageUtils.extract_page_numbers_from_string(page_num_str)
            
            if target_page in chunk_pages:
                last_chunk = chunk
        
        return last_chunk
    
    @staticmethod
    def get_page_range_from_chunks(chunks: List) -> tuple:
        """计算语块列表的页码范围，返回(最小页码, 最大页码)"""
        
        all_pages = []
        for chunk in chunks:
            # 兼容SearchResult对象和字典格式
            if hasattr(chunk, 'page_num'):
                page_num_str = chunk.page_num
            else:
                page_num_str = chunk['_source'].get('page_num', '')
            pages = PageUtils.extract_page_numbers_from_string(page_num_str)
            all_pages.extend(pages)
        
        if all_pages:
            return min(all_pages), max(all_pages)
        else:
            return None, None
    
    @staticmethod
    def get_chunk_id_range_from_pages(all_chunks: List, start_page: int, end_page: int) -> tuple:
        """根据页码范围获取对应的chunk_id范围"""
        
        min_chunk_id = None
        max_chunk_id = None
        
        for chunk in all_chunks:
            # 兼容SearchResult对象和字典格式
            if hasattr(chunk, 'page_num'):
                page_num_str = chunk.page_num
            else:
                page_num_str = chunk['_source'].get('page_num', '')
            chunk_pages = PageUtils.extract_page_numbers_from_string(page_num_str)
            
            if chunk_pages:
                chunk_min_page = min(chunk_pages)
                chunk_max_page = max(chunk_pages)
                
                # 检查是否与目标页码范围有交集
                if chunk_max_page >= start_page and chunk_min_page <= end_page:
                    # 兼容SearchResult对象和字典格式
                    if hasattr(chunk, 'chunk_id'):
                        chunk_id = chunk.chunk_id
                    else:
                        chunk_id = chunk['_source']['chunk_id']
                    
                    if min_chunk_id is None or chunk_id < min_chunk_id:
                        min_chunk_id = chunk_id
                    if max_chunk_id is None or chunk_id > max_chunk_id:
                        max_chunk_id = chunk_id
        
        return min_chunk_id, max_chunk_id