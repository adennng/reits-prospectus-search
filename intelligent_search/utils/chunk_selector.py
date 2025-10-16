#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语块选择器
使用LLM从候选语块中选择最相关的内容
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from searchers.base_searcher import SearchResult
from utils.llm_utils import LLMUtils


class ChunkSelector:
    """语块选择器，使用LLM选择最相关的语块"""
    
    def __init__(self, llm_client, llm_model):
        """初始化语块选择器"""
        self.llm_client = llm_client
        self.llm_model = llm_model
        self._last_selection_note: Optional[str] = None
        print("[ChunkSelector] 语块选择器初始化完成")
    
    def select_best_chunk(
        self,
        search_info: str,
        candidate_results: List[SearchResult],
        all_chunks: List[SearchResult],
        expand_context: bool = True,
        intent: str = "content"
    ) -> Optional[SearchResult]:
        """
        从候选语块中选择最佳语块
        
        Args:
            search_info: 检索需求描述
            candidate_results: 候选语块列表
            all_chunks: 该文件的所有语块（用于扩展上下文）
            expand_context: 是否扩展上下文
            
        Returns:
            Optional[SearchResult]: 最佳语块，如果选择失败则返回None
        """
        
        if not candidate_results:
            print("[ChunkSelector] 候选语块为空")
            self._last_selection_note = "未提供候选语块" if intent == "title" else None
            return None
        
        if len(candidate_results) == 1:
            print("[ChunkSelector] 只有一个候选语块，直接返回")
            self._last_selection_note = None
            return candidate_results[0]
        
        if intent != "title":
            print("[ChunkSelector] 非标题意图无需LLM筛选，返回排序首个候选语块")
            self._last_selection_note = None
            return candidate_results[0]
        
        print(f"[ChunkSelector] 开始选择最佳语块，候选数量: {len(candidate_results)}，意图={intent}")
        self._last_selection_note = None
        
        try:
            expanded_candidates = self._expand_candidates(
                candidate_results,
                all_chunks,
                expand_context
            )
            
            prompt = self._build_selection_prompt(
                search_info=search_info,
                expanded_candidates=expanded_candidates,
                intent=intent
            )
            
            selection_result = self._call_llm_for_selection(prompt)
            selected_index, no_match = self._parse_selection_result(
                selection_result,
                len(candidate_results)
            )
            
            if no_match:
                self._last_selection_note = "未检索到目标标题所在文本块"
                return None
            
            if selected_index is not None:
                selected_chunk = candidate_results[selected_index]
                self._last_selection_note = None
                print(f"[ChunkSelector] 选择了第{selected_index+1}个语块: chunk_id={selected_chunk.chunk_id}")
                return selected_chunk
            
            print("[ChunkSelector] LLM选择失败，返回第一个候选语块")
            return candidate_results[0]
                
        except Exception as e:
            print(f"[ChunkSelector] 语块选择过程中出错: {e}")
            self._last_selection_note = None
            return candidate_results[0]  # 出错时返回第一个
    
    def _expand_candidates(
        self,
        candidate_results: List[SearchResult],
        all_chunks: List[SearchResult],
        expand_context: bool
    ) -> List[Dict[str, Any]]:
        """扩展候选语块，提供更多上下文"""
        
        if not expand_context:
            # 不扩展，直接使用原始文本
            return [
                {
                    'index': i,
                    'chunk_id': result.chunk_id,
                    'original_text': result.text,
                    'expanded_text': result.text,
                    'page_num': result.page_num
                }
                for i, result in enumerate(candidate_results)
            ]
        
        # 创建chunk_id到SearchResult的映射
        chunk_map = {chunk.chunk_id: chunk for chunk in all_chunks}
        
        expanded_candidates = []
        
        for i, candidate in enumerate(candidate_results):
            chunk_id = candidate.chunk_id
            
            # 获取前一个和后一个语块
            prev_chunk = chunk_map.get(chunk_id - 1)
            next_chunk = chunk_map.get(chunk_id + 1)
            
            # 拼接扩展文本
            expanded_parts = []
            
            if prev_chunk:
                expanded_parts.append(f"[前文]{prev_chunk.text}")
            
            expanded_parts.append(f"[目标]{candidate.text}")
            
            if next_chunk:
                expanded_parts.append(f"[后文]{next_chunk.text}")
            
            expanded_text = "\n".join(expanded_parts)
            
            expanded_candidates.append({
                'index': i,
                'chunk_id': chunk_id,
                'original_text': candidate.text,
                'expanded_text': expanded_text,
                'page_num': candidate.page_num
            })
        
        print(f"[ChunkSelector] 完成候选语块扩展，平均长度: {sum(len(c['expanded_text']) for c in expanded_candidates) // len(expanded_candidates)} 字符")
        
        return expanded_candidates
    
    def _build_selection_prompt(
        self,
        search_info: str,
        expanded_candidates: List[Dict[str, Any]],
        intent: str
    ) -> str:
        """构建LLM选择prompt"""
        
        # 构建候选语块信息
        candidates_info = []
        for candidate in expanded_candidates:
            chunk_id = candidate['chunk_id']
            page_num = candidate['page_num']
            expanded_text = candidate['expanded_text']
            
            # 限制文本长度，避免超出LLM限制
            if len(expanded_text) > 1000:
                display_text = expanded_text[:1000] + "..."
            else:
                display_text = expanded_text
            
            candidate_info = (
                f"选项{candidate['index']+1} (chunk_id={chunk_id}, page_num={page_num}):\n"
                f"{display_text}"
            )
            candidates_info.append(candidate_info)
        
        candidates_text = "\n\n".join(candidates_info)
        
        prompt = f"""请从以下候选章节标题语块中选择最符合检索需求的文本块。

检索需求：{search_info}
系统识别的检索意图：{intent}
请你在如下文本块中找出为目标标题所在正文的文本块，请遵循：
- 请选择包含完整章节标题的语块，且后续内容明显是该章节的正文内容；
- 若某文本块虽然包含目标标题的字样，但是仅是对其的引用，则不考虑；
- 如未出现满足条件的语块，请将最终回答中的“最佳选择”填写为“未检索到目标标题所在文本块”，并在理由中说明未命中的原因。

候选文本块：
{candidates_text}

请仔细分析每个选项，仅输出以下JSON：
{{
    "最佳选择": "选项X",
    "选择理由": "详细说明为什么选择这个选项",
    "置信度": "高/中/低"
}}

其中X为选项编号（1、2、3等）。不能输出除JSON之外的任何文字；若未命中标题，也请输出上述JSON，并将“最佳选择”设为“未检索到目标标题所在文本块”。"""
        
        return prompt
    
    def _call_llm_for_selection(self, prompt: str) -> str:
        """调用LLM进行语块选择"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0  # 确保选择的一致性
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # 显示LLM响应预览
            response_preview = raw_response.replace('\n', ' ')[:200]
            print(f"[ChunkSelector] LLM选择响应: {response_preview}...")
            
            return raw_response
            
        except Exception as e:
            print(f"[ChunkSelector] LLM调用失败: {e}")
            return ""
    
    def _parse_selection_result(
        self,
        raw_response: str,
        total_candidates: int
    ) -> Tuple[Optional[int], bool]:
        """解析LLM选择结果，返回(索引, 是否无匹配)"""
        
        try:
            result = LLMUtils.parse_llm_json_response(raw_response)
            
            if not result:
                print("[ChunkSelector] LLM响应JSON解析失败")
                return None, False
            
            selection = (result.get("最佳选择") or "").strip()
            reason = result.get("选择理由", "")
            confidence = result.get("置信度", "")
            
            print("[ChunkSelector] LLM选择结果:")
            print(f"  - 选择: {selection}")
            print(f"  - 理由: {reason}")
            print(f"  - 置信度: {confidence}")
            
            if selection == "未检索到目标标题所在文本块":
                print("[ChunkSelector] LLM判断未找到匹配的标题文本块")
                return None, True
            
            import re
            match = re.search(r'选项(\d+)', selection)
            if match:
                option_num = int(match.group(1))
                index = option_num - 1
                if 0 <= index < total_candidates:
                    return index, False
            
            print(f"[ChunkSelector] 无法解析选项编号: {selection}")
            return None, False
            
        except Exception as e:
            print(f"[ChunkSelector] 解析选择结果失败: {e}")
            return None, False
    
    @property
    def last_selection_note(self) -> Optional[str]:
        """返回最新的LLM选择提示信息"""
        return self._last_selection_note

    def select_best_chunks_batch(
        self,
        search_info: str,
        candidate_groups: List[List[SearchResult]],
        all_chunks: List[SearchResult],
        intent: str = "content"
    ) -> List[Optional[SearchResult]]:
        """
        批量选择最佳语块
        
        Args:
            search_info: 检索需求描述
            candidate_groups: 候选语块组列表
            all_chunks: 所有语块
            
        Returns:
            List[Optional[SearchResult]]: 每组的最佳语块列表
        """
        
        results = []
        for i, candidates in enumerate(candidate_groups):
            print(f"[ChunkSelector] 处理第{i+1}组候选语块...")
            best_chunk = self.select_best_chunk(
                search_info=search_info,
                candidate_results=candidates,
                all_chunks=all_chunks,
                intent=intent
            )
            results.append(best_chunk)
        
        return results