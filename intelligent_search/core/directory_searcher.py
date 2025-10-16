#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
目录检索模块
负责招募说明书目录内容的检索和识别
"""

import sys
import os
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from db_config import get_elasticsearch_config
from utils.llm_utils import LLMUtils
from utils.page_utils import PageUtils


class DirectorySearcher:
    """目录检索类"""
    
    def __init__(self, llm_client, llm_model):
        """初始化目录检索器"""
        # ES连接配置
        self.es_config = get_elasticsearch_config()
        self.es = Elasticsearch(
            [f"{self.es_config['scheme']}://{self.es_config['host']}:{self.es_config['port']}"],
            basic_auth=(self.es_config['username'], self.es_config['password']),
            verify_certs=False,
            ssl_show_warn=False
        )
        
        # LLM客户端
        self.llm_client = llm_client
        self.llm_model = llm_model
        
        # ES索引配置
        self.es_index = "reits_announcements"
        self.chunks_before = 0  # 目录块向前扩展
        self.chunks_after = 7   # 目录块向后扩展
        
        print("[DirectorySearcher] 目录检索器初始化完成")
    
    def get_directory_content(self, fund_code: str, source_file: str) -> Dict[str, Any]:
        """获取目录内容"""
        
        print(f"[DirectorySearcher] 开始获取目录内容...")
        
        try:
            # 1. 从ES获取该文件的所有语块，按chunk_id升序
            hits = self._get_file_chunks_from_es(fund_code, source_file)
            if not hits:
                return self._create_error_result("ES中未找到该文件的语块数据")
            
            print(f"[DirectorySearcher] 从ES获取到 {len(hits)} 个语块")
            
            # 2. 确保按chunk_id升序排序
            hits.sort(key=lambda h: h['_source']['chunk_id'])
            
            # 3. 构建chunk_id到text的映射，便于后续扩展
            id2text = {h['_source']['chunk_id']: h['_source']['text'] for h in hits}
            
            # 4. 筛选包含"目"和"录"的候选语块
            candidates = [h for h in hits if ("目" in h['_source']['text'] and "录" in h['_source']['text'])]
            
            print(f"[DirectorySearcher] 关键词筛选后候选语块 {len(candidates)} 个")
            
            if not candidates:
                return self._create_error_result("未找到包含目录关键词的语块")
            
            # 5. 使用LLM逐个判断哪个是真正的目录语块
            directory_chunk = None
            llm_check_times = 0
            
            for chunk in candidates:
                llm_check_times += 1
                chunk_id = chunk['_source']['chunk_id']
                preview = chunk['_source']['text'][:200].replace("\n", " ")
                print(f"[DirectorySearcher] 检查候选chunk {chunk_id}: {preview}...")
                
                # 扩展当前语块+后2块用于LLM判断
                combined_text = (
                    chunk['_source']['text'] + 
                    id2text.get(chunk_id + 1, "") + 
                    id2text.get(chunk_id + 2, "")
                )
                
                if self._is_directory_chunk_by_llm(combined_text):
                    directory_chunk = chunk
                    print(f"[DirectorySearcher] 确定目录语块: chunk {chunk_id}")
                    break
            
            print(f"[DirectorySearcher] 共进行 {llm_check_times} 次LLM目录判断")
            
            if directory_chunk is None:
                return self._create_error_result("LLM未能识别出目录语块")
            
            # 6. 扩展目录语块获取完整目录内容
            dir_chunk_id = directory_chunk['_source']['chunk_id']
            start_chunk_id = dir_chunk_id - self.chunks_before  # 向前扩展0块
            end_chunk_id = dir_chunk_id + self.chunks_after     # 向后扩展7块
            
            # 筛选扩展范围内的语块
            expanded_chunks = [h for h in hits if start_chunk_id <= h['_source']['chunk_id'] <= end_chunk_id]
            
            if not expanded_chunks:
                return self._create_error_result("扩展目录语块失败")
            
            # 7. 拼接完整目录文本
            directory_text = "".join(h['_source']['text'] for h in expanded_chunks)
            
            # 8. 计算页码信息
            start_page, end_page = PageUtils.calculate_page_range(expanded_chunks)
            actual_start_chunk_id = expanded_chunks[0]['_source']['chunk_id']
            actual_end_chunk_id = expanded_chunks[-1]['_source']['chunk_id']
            
            print(f"[DirectorySearcher] 目录内容获取成功，文本长度: {len(directory_text)}")
            print(f"[DirectorySearcher] 页码范围: {start_page}-{end_page}, chunk范围: {actual_start_chunk_id}-{actual_end_chunk_id}")
            
            return self._create_success_result(
                source_file=source_file,
                content=directory_text,
                start_page=start_page,
                end_page=end_page,
                start_chunk_id=actual_start_chunk_id,
                end_chunk_id=actual_end_chunk_id
            )
            
        except Exception as e:
            error_msg = f"获取目录内容时发生异常: {str(e)}"
            print(f"[DirectorySearcher] {error_msg}")
            return self._create_error_result(error_msg)
    
    def _get_file_chunks_from_es(self, fund_code: str, source_file: str) -> List:
        """从ES获取指定文件的所有语块"""
        
        def _build_query(field):
            """构建ES查询体"""
            return {
                "size": 10000,
                "sort": [{"chunk_id": "asc"}],
                "_source": ["chunk_id", "text", "page_num", "global_id"],
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"fund_code": fund_code}},
                            {"term": {field: source_file}}
                        ]
                    }
                }
            }
        
        try:
            # 首先尝试使用keyword字段查询
            hits = self.es.search(index=self.es_index, body=_build_query("source_file.keyword"))['hits']['hits']
            
            # 如果没有结果，尝试使用普通字段查询
            if not hits:
                hits = self.es.search(index=self.es_index, body=_build_query("source_file"))['hits']['hits']
            
            print(f"[DirectorySearcher] ES查询返回 {len(hits)} 个语块")
            return hits
            
        except Exception as e:
            print(f"[DirectorySearcher] ES查询失败: {e}")
            return []
    
    def _is_directory_chunk_by_llm(self, text_snippet: str) -> bool:
        """使用LLM判断文本片段是否为目录"""
        
        # 创建目录判定Prompt
        prompt = LLMUtils.create_directory_check_prompt(text_snippet)
        
        try:
            preview = text_snippet[:400].replace('\n', ' ')
            print(f"[DirectorySearcher] LLM目录判断输入: {preview}...")
            
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            raw_response = response.choices[0].message.content.strip()
            response_preview = raw_response.replace('\n', ' ')[:200]
            print(f"[DirectorySearcher] LLM返回: {response_preview}...")
            
            # 解析LLM返回的JSON
            result = LLMUtils.parse_llm_json_response(raw_response)
            is_directory = result.get("是目录")
            
            if is_directory:
                return LLMUtils.normalize_yes_value(is_directory)
            else:
                # 兼容直接返回"是"/"否"的情况
                return LLMUtils.normalize_yes_value(raw_response.strip())
                
        except Exception as e:
            print(f"[DirectorySearcher] LLM目录判断异常: {e}")
            return False
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            "success": False,
            "source_file": None,
            "content": None,
            "start_page": None,
            "end_page": None,
            "start_chunk_id": None,
            "end_chunk_id": None,
            "error": error_msg
        }
    
    def _create_success_result(
        self,
        source_file: str,
        content: str,
        start_page: int,
        end_page: int,
        start_chunk_id: int,
        end_chunk_id: int
    ) -> Dict[str, Any]:
        """创建成功结果"""
        return {
            "success": True,
            "source_file": source_file,
            "content": content,
            "start_page": start_page,
            "end_page": end_page,
            "start_chunk_id": start_chunk_id,
            "end_chunk_id": end_chunk_id,
            "error": None
        }