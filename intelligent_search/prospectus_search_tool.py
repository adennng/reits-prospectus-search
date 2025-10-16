#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
招募说明书智能检索工具 - 重构版本

实现让LLM像人类一样查阅招募说明书的检索功能，支持：
1. 目录查询
2. 章节定位检索  
3. 范围限制检索
4. 智能文本扩展
"""

import sys
import os
from typing import Dict, Any, List, Optional

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入配置文件
from db_config import get_vector_db_config
from model_config import MODEL_CONFIG

# 导入LLM相关库
from openai import OpenAI

# 导入重构后的模块
from core.file_manager import FileManager
from core.directory_searcher import DirectorySearcher
from utils.page_utils import PageUtils
from utils.chunk_utils import ChunkUtils
from utils.chunk_selector import ChunkSelector
from searchers import KeywordSearcher, VectorSearcher, HybridSearcher, SearchResult


# 默认的检索意图到检索模式的映射，可根据需要手动调整
DEFAULT_INTENT_MODE_MAP = {
    "title": "keyword",
    "content": "hybrid",
}


class ProspectusSearchTool:
    """
    招募说明书智能检索工具 - 重构版本
    
    支持LLM多轮调用，逐步定位和获取招募说明书中的特定信息
    """
    
    def __init__(self):
        """初始化检索工具"""
        print("[ProspectusSearchTool] 招募说明书智能检索工具初始化开始...")
        
        # 配置信息
        self.vector_config = get_vector_db_config()
        
        # 初始化LLM客户端（使用ali的deepseek-v3）
        llm_config = MODEL_CONFIG["ali"]["deepseek-v3"]
        self.llm_client = OpenAI(
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"]
        )
        self.llm_model = llm_config["model"]
        print(f"[ProspectusSearchTool] LLM客户端初始化成功，使用模型: {self.llm_model}")
        
        # 初始化各个功能模块
        self.file_manager = FileManager()
        self.directory_searcher = DirectorySearcher(self.llm_client, self.llm_model)
        self.chunk_selector = ChunkSelector(self.llm_client, self.llm_model)
        
        # 初始化检索器（懒加载，在使用时才创建）
        self._keyword_searcher = None
        self._vector_searcher = None
        self._hybrid_searcher = None

        # 检索意图到检索模式的映射，可根据业务需要在此调整
        self.intent_mode_map = DEFAULT_INTENT_MODE_MAP.copy()

        print("[ProspectusSearchTool] 招募说明书智能检索工具初始化完成")
    
    def search_prospectus(
        self,
        fund_code: str,                           # 基金代码（必填）
        search_info: str,                         # 检索信息（可为空字符串）
        is_expansion: bool = False,               # 是否扩募（默认False）
        start_page: Optional[int] = None,         # 起始页码
        end_page: Optional[int] = None,           # 截止页码
        start_chunk_id: Optional[int] = None,     # 起始chunk_id
        end_chunk_id: Optional[int] = None,       # 截止chunk_id
        expand_before: int = 0,                   # 向前扩展文本块数量
        expand_after: int = 0                     # 向后扩展文本块数量
    ) -> Dict[str, Any]:
        """
        执行招募说明书智能检索

        Args:
            fund_code: 基金代码，如 "180301.SZ"
            search_info: 检索信息，可以是"目录"、具体内容描述，或空字符串表示直接按范围取文
            is_expansion: 是否查询扩募招募说明书，False为首发
            start_page: 起始页码，用于范围限制
            end_page: 截止页码，用于范围限制
            start_chunk_id: 起始chunk_id，用于范围限制
            end_chunk_id: 截止chunk_id，用于范围限制
            expand_before: 向前扩展的文本块数量
            expand_after: 向后扩展的文本块数量
            检索模式会根据识别出的检索意图自动选择，可通过 `self.intent_mode_map` 调整映射

        Returns:
            Dict[str, Any]: 根据检索意图返回不同结构：
                - 标题检索：返回单条正文内容及其页码、chunk 范围。
                - 内容检索：返回多条正文结果列表，每条包含文本与位置信息。
        """

        print(f"[ProspectusSearchTool] 开始检索: 基金={fund_code}, 内容={search_info[:50]}...")

        intent_for_error = self._infer_intent_for_error(search_info)

        # 参数验证
        validation_result = self._validate_parameters(
            fund_code, search_info, start_page, end_page,
            start_chunk_id, end_chunk_id
        )
        if not validation_result["valid"]:
            return self._create_error_result(validation_result["error"], intent=intent_for_error)

        try:
            # 第一步：确定目标招募说明书文件名
            source_file = self.file_manager.determine_prospectus_file(fund_code, is_expansion)
            if not source_file:
                error_msg = f"未找到基金 {fund_code} 的{'扩募' if is_expansion else '首发'}招募说明书"
                return self._create_error_result(error_msg, intent=intent_for_error)

            print(f"[ProspectusSearchTool] 确定目标文件: {source_file}")

            # 第二步：根据检索信息类型进行处理
            if search_info == "目录":
                # 特殊处理：返回完整目录信息
                return self.directory_searcher.get_directory_content(fund_code, source_file)

            # 一般文本检索
            return self._search_general_content(
                fund_code=fund_code,
                source_file=source_file,
                search_info=search_info,
                start_page=start_page,
                end_page=end_page,
                start_chunk_id=start_chunk_id,
                end_chunk_id=end_chunk_id,
                expand_before=expand_before,
                expand_after=expand_after
            )

        except Exception as e:
            error_msg = f"检索过程中发生错误: {str(e)}"
            print(f"[ProspectusSearchTool] {error_msg}")
            return self._create_error_result(error_msg, intent=intent_for_error)
    
    def _validate_parameters(
        self, 
        fund_code: str, 
        search_info: str,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        start_chunk_id: Optional[int] = None,
        end_chunk_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """参数验证"""
        
        # 基金代码验证
        if not fund_code or not fund_code.strip():
            return {"valid": False, "error": "基金代码不能为空"}
        
        # 检索信息验证
        if search_info is None:
            return {"valid": False, "error": "检索信息不能为空"}
        
        # 页码范围验证
        if start_page is not None and end_page is not None:
            if start_page > end_page:
                return {"valid": False, "error": "起始页码不能大于截止页码"}
        
        # chunk_id范围验证
        if start_chunk_id is not None and end_chunk_id is not None:
            if start_chunk_id > end_chunk_id:
                return {"valid": False, "error": "起始chunk_id不能大于截止chunk_id"}
        
        return {"valid": True}
    
    def _infer_intent_for_error(self, search_info: Optional[str]) -> str:
        """根据检索信息推断错误响应结构"""
        if not search_info or search_info == "目录":
            return "content"
        try:
            parsed = self._parse_search_intent(search_info)
            return parsed.get("intent", "content")
        except Exception:
            return "content"

    def _search_general_content(
        self,
        fund_code: str,
        source_file: str,
        search_info: str,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        start_chunk_id: Optional[int] = None,
        end_chunk_id: Optional[int] = None,
        expand_before: int = 0,
        expand_after: int = 0
    ) -> Dict[str, Any]:
        """执行一般内容检索"""

        print("[ProspectusSearchTool] 执行一般内容检索")

        intent = "content"

        try:
            parsed_intent = self._parse_search_intent(search_info)
            intent = parsed_intent['intent']
            normalized_query = parsed_intent['query']
            print(
                f"[ProspectusSearchTool] 检索意图: {intent}, 归一化查询: {normalized_query[:80]}"
            )

            search_mode = self._resolve_search_mode(intent)
            print(f"[ProspectusSearchTool] 根据意图选择检索模式: {search_mode}")

            # 1. 获取文件所有语块（用于范围限制和语块扩展）
            all_chunks = self._get_all_file_chunks(fund_code, source_file)
            if not all_chunks:
                return self._create_error_result("未能获取文件语块数据", intent=intent)

            print(f"[ProspectusSearchTool] 获取文件语块 {len(all_chunks)} 个")

            # 2. 根据范围参数计算候选区间
            chunk_range_limits = None
            if any(value is not None for value in [start_page, end_page, start_chunk_id, end_chunk_id]):
                range_chunks = ChunkUtils.apply_range_limitations(
                    all_chunks,
                    start_page,
                    end_page,
                    start_chunk_id,
                    end_chunk_id
                )

                if not range_chunks:
                    return self._create_error_result("指定范围内无内容", intent=intent)

                chunk_range_limits = ChunkUtils.get_chunk_id_range_from_chunks(range_chunks)
                print(
                    f"[ProspectusSearchTool] 检索范围限定 chunk_id: {chunk_range_limits[0]}-{chunk_range_limits[1]}"
                )

            # 3. 如果检索信息为空，直接返回范围内文本
            if not normalized_query.strip():
                print("[ProspectusSearchTool] 检索信息为空，直接按范围获取文本内容")
                return self._get_range_content(
                    all_chunks,
                    start_page,
                    end_page,
                    start_chunk_id,
                    end_chunk_id,
                    source_file
                )

            # 4. 执行检索获取候选语块
            candidate_results = self._execute_search(
                normalized_query,
                fund_code,
                source_file,
                search_mode,
                chunk_range_limits,
                intent
            )

            if not candidate_results:
                return self._create_error_result("未找到匹配的内容", intent=intent)

            print(f"[ProspectusSearchTool] 获得候选语块 {len(candidate_results)} 个")
            self._log_candidate_chunks(
                label=f"{search_mode}检索候选（初始）",
                chunks=candidate_results
            )

            # 5. 应用范围限制（再次保证结果在范围内）
            if start_page or end_page or start_chunk_id or end_chunk_id:
                candidate_results = self._apply_range_filter(
                    candidate_results, start_page, end_page,
                    start_chunk_id, end_chunk_id
                )
                print(f"[ProspectusSearchTool] 范围限制后保留 {len(candidate_results)} 个候选语块")
                self._log_candidate_chunks(
                    label="范围限制后的候选",
                    chunks=candidate_results
                )

            if not candidate_results:
                return self._create_error_result("应用范围限制后无匹配结果", intent=intent)

            if intent == "title":
                best_chunk = self.chunk_selector.select_best_chunk(
                    search_info=search_info,
                    candidate_results=candidate_results,
                    all_chunks=all_chunks,
                    expand_context=True,
                    intent=intent
                )

                if best_chunk is None:
                    note = getattr(self.chunk_selector, "last_selection_note", None)
                    message = note or "未检索到目标标题所在文本块"
                    print(f"[ProspectusSearchTool] LLM未选出标题语块: {message}")
                    return self._create_error_result(message, intent=intent)

                print(f"[ProspectusSearchTool] LLM选择最佳语块: chunk_id={best_chunk.chunk_id}")
                expanded_results = self._prepare_expanded_results(
                    [best_chunk],
                    all_chunks,
                    expand_before,
                    expand_after
                )
            else:
                expanded_results = self._prepare_expanded_results(
                    candidate_results,
                    all_chunks,
                    expand_before,
                    expand_after
                )

            if not expanded_results:
                return self._create_error_result("语块扩展后无有效内容", intent=intent)

            print(f"[ProspectusSearchTool] 一般内容检索成功，返回 {len(expanded_results)} 条结果")

            if intent == "title":
                return self._create_title_success_result(
                    source_file=source_file,
                    results=expanded_results
                )
            return self._create_content_success_result(
                source_file=source_file,
                results=expanded_results
            )

        except Exception as e:
            error_msg = f"一般内容检索失败: {str(e)}"
            print(f"[ProspectusSearchTool] {error_msg}")
            return self._create_error_result(error_msg, intent=intent)

    def _get_all_file_chunks(self, fund_code: str, source_file: str) -> List[SearchResult]:
        """获取文件的所有语块"""
        try:
            # 使用关键词检索器获取所有语块
            searcher = self._get_keyword_searcher()
            chunks = searcher.get_file_chunks(fund_code, source_file, sort_by_chunk_id=True)
            return chunks
        except Exception as e:
            print(f"[ProspectusSearchTool] 获取文件语块失败: {e}")
            return []
    
    def _execute_search(
        self, 
        search_info: str, 
        fund_code: str, 
        source_file: str, 
        search_mode: str,
        chunk_range: Optional[tuple],
        intent: str
    ) -> List[SearchResult]:
        """执行指定模式的检索"""
        try:
            if search_mode == "keyword":
                searcher = self._get_keyword_searcher()
                return searcher.search(
                    search_info,
                    fund_code,
                    source_file,
                    top_k=10,
                    chunk_range=chunk_range,
                    intent=intent
                )
            elif search_mode == "vector":
                searcher = self._get_vector_searcher()
                return searcher.search(
                    search_info,
                    fund_code,
                    source_file,
                    top_k=10,
                    chunk_range=chunk_range,
                    intent=intent
                )
            elif search_mode == "hybrid":
                searcher = self._get_hybrid_searcher()
                return searcher.search(
                    search_info,
                    fund_code,
                    source_file,
                    top_k=10,
                    chunk_range=chunk_range,
                    intent=intent
                )
            else:
                print(f"[ProspectusSearchTool] 未知的检索模式: {search_mode}")
                return []
        except Exception as e:
            print(f"[ProspectusSearchTool] 执行检索失败: {e}")
            return []
    
    def _apply_range_filter(
        self,
        candidate_results: List[SearchResult],
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        start_chunk_id: Optional[int] = None,
        end_chunk_id: Optional[int] = None
    ) -> List[SearchResult]:
        """应用范围过滤"""
        try:
            return ChunkUtils.apply_range_limitations(
                candidate_results, start_page, end_page, 
                start_chunk_id, end_chunk_id
            )
        except Exception as e:
            print(f"[ProspectusSearchTool] 应用范围过滤失败: {e}")
            return candidate_results
    
    def _prepare_expanded_results(
        self,
        candidate_results: List[SearchResult],
        all_chunks: List[SearchResult],
        expand_before: int,
        expand_after: int
    ) -> List[Dict[str, Any]]:
        """按候选逐条扩展并生成结果列表"""
        expanded_results: List[Dict[str, Any]] = []

        for order, candidate in enumerate(candidate_results, start=1):
            try:
                expanded_chunks = ChunkUtils.expand_chunks(
                    [candidate], all_chunks, expand_before, expand_after
                )
            except Exception as exc:
                print(f"[ProspectusSearchTool] 扩展语块失败: {exc}")
                expanded_chunks = []

            if not expanded_chunks:
                expanded_chunks = [candidate]

            result_entry = self._build_expanded_entry(
                expanded_chunks=expanded_chunks,
                base_chunk=candidate,
                order=order
            )
            expanded_results.append(result_entry)

        return expanded_results

    def _build_expanded_entry(
        self,
        expanded_chunks: List[SearchResult],
        base_chunk: Optional[SearchResult],
        order: int
    ) -> Dict[str, Any]:
        """构造单条扩展后的检索结果"""
        if not expanded_chunks and base_chunk is not None:
            expanded_chunks = [base_chunk]

        # 按 chunk_id 排序，确保范围计算准确
        sorted_chunks = sorted(
            expanded_chunks,
            key=lambda chunk: getattr(chunk, 'chunk_id', 0)
        )

        merged_text = ChunkUtils.merge_chunks_text(sorted_chunks)
        start_page, end_page = PageUtils.get_page_range_from_chunks(sorted_chunks)
        start_chunk_id, end_chunk_id = ChunkUtils.get_chunk_id_range_from_chunks(sorted_chunks)
        chunk_ids = [chunk.chunk_id for chunk in sorted_chunks if hasattr(chunk, 'chunk_id')]

        base_chunk_id = getattr(base_chunk, 'chunk_id', None) if base_chunk else None
        base_page_num = getattr(base_chunk, 'page_num', '') if base_chunk else ''

        page_range = [start_page, end_page] if start_page is not None and end_page is not None else None
        chunk_range = [start_chunk_id, end_chunk_id] if start_chunk_id is not None and end_chunk_id is not None else None

        return {
            'order': order,
            'base_chunk_id': base_chunk_id,
            'base_page_num': base_page_num,
            'text': merged_text,
            'start_page': start_page,
            'end_page': end_page,
            'start_chunk_id': start_chunk_id,
            'end_chunk_id': end_chunk_id,
            'page_range': page_range,
            'chunk_range': chunk_range,
            'chunk_ids': chunk_ids
        }

    def _get_range_content(
        self,
        all_chunks: List[SearchResult],
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        start_chunk_id: Optional[int] = None,
        end_chunk_id: Optional[int] = None,
        source_file: str = ""
    ) -> Dict[str, Any]:
        """获取指定范围内的内容（当检索信息为空时）"""
        try:
            # 应用范围限制
            range_chunks = ChunkUtils.apply_range_limitations(
                all_chunks, start_page, end_page, start_chunk_id, end_chunk_id
            )

            if not range_chunks:
                return self._create_error_result("指定范围内无内容", intent="content")

            self._log_candidate_chunks(
                label="范围文本语块",
                chunks=range_chunks,
                limit=10
            )

            expanded_entry = self._build_expanded_entry(
                expanded_chunks=range_chunks,
                base_chunk=range_chunks[0] if range_chunks else None,
                order=1
            )

            return self._create_content_success_result(
                source_file=source_file,
                results=[expanded_entry]
            )
        except Exception as e:
            return self._create_error_result(f"获取范围内容失败: {str(e)}", intent="content")
    
    def _parse_search_intent(self, search_info: str) -> Dict[str, str]:
        """解析检索意图和真实查询内容"""

        if search_info is None:
            return {"intent": "content", "query": ""}

        raw = search_info.strip()
        if not raw:
            return {"intent": "content", "query": ""}

        prefix_map = {
            "title": ["章节标题检索：", "章节标题检索:"],
            "content": ["内容检索：", "内容检索:"]
        }

        for intent_key, prefixes in prefix_map.items():
            for prefix in prefixes:
                if raw.startswith(prefix):
                    return {"intent": intent_key, "query": raw[len(prefix):].strip()}

        return {"intent": "content", "query": raw}

    def _resolve_search_mode(self, intent: str) -> str:
        """根据检索意图选择检索模式"""

        mode = self.intent_mode_map.get(intent)
        if mode is None:
            mode = self.intent_mode_map.get("content", "hybrid")

        if mode not in {"keyword", "vector", "hybrid"}:
            print(
                f"[ProspectusSearchTool] 意图 {intent} 映射到非法模式 {mode}，改用默认 hybrid"
            )
            return "hybrid"

        return mode

    def _get_keyword_searcher(self) -> KeywordSearcher:
        """获取关键词检索器（懒加载）"""
        if self._keyword_searcher is None:
            self._keyword_searcher = KeywordSearcher()
        return self._keyword_searcher
    
    def _get_vector_searcher(self) -> VectorSearcher:
        """获取向量检索器（懒加载）"""
        if self._vector_searcher is None:
            self._vector_searcher = VectorSearcher()
        return self._vector_searcher
    
    def _get_hybrid_searcher(self) -> HybridSearcher:
        """获取混合检索器（懒加载）"""
        if self._hybrid_searcher is None:
            self._hybrid_searcher = HybridSearcher()
        return self._hybrid_searcher
    
    def close_connections(self):
        """关闭所有连接"""
        try:
            if self._keyword_searcher:
                self._keyword_searcher.close_connection()
            if self._vector_searcher:
                self._vector_searcher.close_connection()
            if self._hybrid_searcher:
                self._hybrid_searcher.close_connection()
            print("[ProspectusSearchTool] 所有连接已关闭")
        except Exception as e:
            print(f"[ProspectusSearchTool] 关闭连接时出错: {e}")
    
    def _create_error_result(self, error_msg: str, intent: str = "content") -> Dict[str, Any]:
        """根据意图构造错误结果"""
        if intent == "title":
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
        return {
            "success": False,
            "source_file": None,
            "error": error_msg,
            "retrieved_count": 0,
            "retrieved_summary": None,
            "results": []
        }

    def _create_title_success_result(
        self,
        source_file: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """构造标题检索成功结果"""
        entry = results[0] if results else {}
        return {
            "success": True,
            "source_file": source_file,
            "content": entry.get("text"),
            "start_page": entry.get("start_page"),
            "end_page": entry.get("end_page"),
            "start_chunk_id": entry.get("start_chunk_id"),
            "end_chunk_id": entry.get("end_chunk_id"),
            "error": None
        }

    def _create_content_success_result(
        self,
        source_file: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """构造内容检索成功结果"""
        formatted_results: List[Dict[str, Any]] = [
            {
                "text": item.get("text"),
                "start_page": item.get("start_page"),
                "end_page": item.get("end_page"),
                "start_chunk_id": item.get("start_chunk_id"),
                "end_chunk_id": item.get("end_chunk_id")
            }
            for item in results
        ]
        retrieved_count = len(formatted_results)
        summary = (
            f"共检索到 {retrieved_count} 个文本信息" if retrieved_count else "未检索到文本信息"
        )
        return {
            "success": True,
            "source_file": source_file,
            "error": None,
            "retrieved_count": retrieved_count,
            "retrieved_summary": summary,
            "results": formatted_results
        }

    def _log_candidate_chunks(
        self,
        label: str,
        chunks: List[Any],
        max_preview: int = 400,
        limit: Optional[int] = None
    ) -> None:
        """打印候选或范围语块的调试信息"""

        total = len(chunks)
        print(f"[ProspectusSearchTool] {label}: 共 {total} 个语块")

        if total == 0:
            return

        display_chunks = chunks if limit is None else chunks[:limit]

        for idx, chunk in enumerate(display_chunks, start=1):
            if hasattr(chunk, 'chunk_id'):
                chunk_id = chunk.chunk_id
                page_num = getattr(chunk, 'page_num', '')
                text = getattr(chunk, 'text', '') or ''
                methods = getattr(chunk, 'from_methods', []) or []
            else:
                source = chunk.get('_source', chunk)
                chunk_id = source.get('chunk_id')
                page_num = source.get('page_num', '')
                text = source.get('text', '') or ''
                methods = source.get('from_methods', []) or []

            preview = text.replace('\n', ' ')[:max_preview]
            method_label = ','.join(methods) if methods else '-'
            print(
                f"  - [{idx}/{total}] chunk_id={chunk_id}, page_num={page_num}, 来源={method_label}, 预览={preview}"
            )

        if limit is not None and total > limit:
            print(f"[ProspectusSearchTool] {label}: 仅展示前 {limit} 个语块，剩余 {total - limit} 个未展开")


# 测试函数
def test_refactored_tool():
    """测试重构后的工具"""
    print("=== 测试重构后的招募说明书智能检索工具 ===")
    
    try:
        # 初始化工具
        tool = ProspectusSearchTool()
        print("✅ 重构后工具初始化成功")
        
        # 测试目录检索
        print("\n🔍 测试目录检索...")
        result = tool.search_prospectus(
            fund_code="180301.SZ",
            search_info="目录",
            is_expansion=False
        )
        
        if result["success"]:
            print("✅ 目录检索成功")
            print(f"📁 源文件: {result['source_file']}")
            print(f"📄 页码范围: {result['start_page']}-{result['end_page']}")
            print(f"🔢 chunk范围: {result['start_chunk_id']}-{result['end_chunk_id']}")
            print(f"📝 内容长度: {len(result['content'])} 字符")
        else:
            print(f"❌ 目录检索失败: {result['error']}")
        
        # 测试一般内容检索（预期失败，因为尚未实现）
        print("\n🔍 测试一般内容检索...")
        result = tool.search_prospectus(
            fund_code="180301.SZ",
            search_info="基金费用",
            is_expansion=False
        )
        
        if result["success"]:
            print("✅ 一般内容检索成功")
        else:
            print(f"📝 一般内容检索预期失败: {result['error']}")
        
        print("\n✅ 重构后工具测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_refactored_tool()
