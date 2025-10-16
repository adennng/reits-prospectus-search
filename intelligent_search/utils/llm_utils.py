#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM相关工具模块
包含LLM调用、JSON解析、结果标准化等功能
"""

from typing import Dict, Any
import json
import re


class LLMUtils:
    """LLM相关工具类"""
    
    @staticmethod
    def parse_llm_json_response(raw_response: str) -> Dict:
        """解析LLM返回的JSON响应"""
        
        # 清理响应文本
        text = raw_response.strip()
        # 移除可能的markdown代码块标记
        text = re.sub(r'^```(?:json)?', '', text)
        text = re.sub(r'```$', '', text).strip()
        
        try:
            return json.loads(text)
        except:
            # 如果JSON解析失败，尝试正则提取
            match = re.search(r'\{[^}]*\}', text)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            return {}
    
    @staticmethod
    def normalize_yes_value(value) -> bool:
        """标准化是/否值"""
        
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        
        value_str = str(value).lower().strip()
        return value_str in ("是", "yes", "true", "1")
    
    @staticmethod
    def create_directory_check_prompt(text_snippet: str) -> str:
        """创建目录判断的LLM prompt"""
        
        prompt = (
            "下面是一段招募说明书文本，请判断该文本是否包含招募说明书的'目录'部分。"
            "如果包含目录，请严格输出 JSON：{\"是目录\":\"是\"}；"
            "如果不包含目录，请严格输出 JSON：{\"是目录\":\"否\"}。"
            "不能输出除 JSON 之外的任何文字；键必须为 '是目录'，值只能是 '是' 或 '否'。\n\n"
            f"文本：\n{text_snippet}"
        )
        
        return prompt
    
    @staticmethod
    def parse_chunk_selection_response(raw_response: str, total_options: int) -> int:
        """解析语块选择响应，返回选中的选项索引（0-based）"""
        
        try:
            data = LLMUtils.parse_llm_json_response(raw_response)
            selection = data.get("最佳选择", "")
            
            # 提取数字
            match = re.search(r'选项(\d+)', selection)
            if match:
                option_num = int(match.group(1))
                # 转换为0-based索引
                index = option_num - 1
                if 0 <= index < total_options:
                    return index
            
            # 如果解析失败，返回第一个选项
            return 0
            
        except:
            # 异常情况下返回第一个选项
            return 0
