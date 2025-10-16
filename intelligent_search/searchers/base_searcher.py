#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础检索类
定义检索工具的统一接口和通用功能
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """检索结果数据类"""
    global_id: str
    chunk_id: int
    source_file: str
    page_num: str
    text: str
    score: float
    fund_code: str = ""
    date: str = ""
    short_name: str = ""
    from_methods: List[str] = None
    
    def __post_init__(self):
        if self.from_methods is None:
            self.from_methods = []


class BaseSearcher(ABC):
    """基础检索类，定义统一接口"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化检索器"""
        self.config = config
        self._connection = None
        self._initialize_connection()
    
    @abstractmethod
    def _initialize_connection(self):
        """初始化数据库连接"""
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        fund_code: Optional[str] = None,
        source_file: Optional[str] = None,
        top_k: int = 10,
        **kwargs
    ) -> List[SearchResult]:
        """执行检索"""
        pass
    
    def _format_search_result(self, raw_result: Dict, score: float, method: str) -> SearchResult:
        """格式化检索结果为统一结构"""
        source = raw_result.get('_source', raw_result)
        
        return SearchResult(
            global_id=source.get('global_id', ''),
            chunk_id=source.get('chunk_id', 0),
            source_file=source.get('source_file', ''),
            page_num=source.get('page_num', ''),
            text=source.get('text', ''),
            score=score,
            fund_code=source.get('fund_code', ''),
            date=source.get('date', ''),
            short_name=source.get('short_name', ''),
            from_methods=[method]
        )
    
    def _build_filters(
        self,
        fund_code: Optional[str] = None,
        source_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """构建过滤条件"""
        filters = {}
        
        if fund_code:
            filters['fund_code'] = fund_code
        
        if source_file:
            filters['source_file'] = source_file
        
        return filters
    
    def close_connection(self):
        """关闭连接"""
        if self._connection:
            try:
                self._connection.close()
                print(f"[{self.__class__.__name__}] 连接已关闭")
            except:
                pass