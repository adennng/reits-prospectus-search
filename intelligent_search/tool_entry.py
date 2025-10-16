"""面向 LLM 的招募说明书检索工具封装"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .prospectus_search_tool import ProspectusSearchTool

# 单例缓存，避免重复初始化造成的资源浪费
_TOOL_INSTANCE: Optional[ProspectusSearchTool] = None

TOOL_NAME = "prospectus_search"

PROSPECTUS_SEARCH_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "根据基金代码检索招募说明书，可查询目录或指定范围的文本内容；"
            "目录检索返回完整目录段落；章节标题检索仅返回单条正文内容及其页码/chunk 范围；"
            "内容检索返回多条正文结果列表，每条包含文本和对应页码/chunk 范围。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fund_code": {
                    "type": "string",
                    "description": "基金代码，例如 180301.SZ"
                },
                "search_info": {
                    "type": "string",
                    "description": "检索的关键词；支持四种方式，一是如果需要获取完整目录信息，则填写“目录”；二是检索包含已知章节标题的文本块，则填写“章节标题检索：目标章节标题”，例如“章节标题检索：第十四部分 基础设施项目基本情况”；三是检索包含特定信息/答案的文本块，则填写“内容检索：需要检索的内容”,例如“内容检索：基金管理费的费率说明”；四是填写信息为空，则会返回限定范围（页码/chunk_id）之间的全部文本块信息”。"
                },
                "is_expansion": {
                    "type": "boolean",
                    "description": "是否检索扩募版招募说明书，默认首发版，如需要检索扩募版招募说明书，则填写True。",
                    "default": False
                },
                "start_page": {
                    "type": "integer",
                    "description": "检索范围限定起始页码（含），如果填写了此项，则检索范围限定为起始页码之后的文本块；如果未填写，则检索范围从第一页开始。",
                    "minimum": 1,
                    "nullable": True
                },
                "end_page": {
                    "type": "integer",
                    "description": "检索范围限定结束页码（含），如果填写了此项，则检索范围限定为截止页码之前的文本块；如果未填写，则检索范围到最后一页。",
                    "minimum": 1,
                    "nullable": True
                },
                "start_chunk_id": {
                    "type": "integer",
                    "description": "检索范围限定起始 chunk_id（含），如果填写了此项，则检索范围限定为起始chunk_id之后的文本块；如果未填写，则检索范围从第一个文本块开始。",
                    "minimum": 0,
                    "nullable": True
                },
                "end_chunk_id": {
                    "type": "integer",
                    "description": "检索范围限定结束 chunk_id（含），如果填写了此项，则检索范围限定为截止chunk_id之前的文本块；如果未填写，则检索范围到最后一个文本块。",
                    "minimum": 0,
                    "nullable": True
                },
                "expand_before": {
                    "type": "integer",
                    "description": "向前追加返回文本块的数量，用于扩展上下文，默认0，如需要向前扩展上文，则填写大于0的整数。",
                    "minimum": 0,
                    "default": 0
                },
                "expand_after": {
                    "type": "integer",
                    "description": "向后追加返回文本块的数量，用于扩展上下文，默认0，如需要向后扩展下文，则填写大于0的整数。",
                    "minimum": 0,
                    "default": 0
                }
            },
            "required": ["fund_code", "search_info"],
            "additionalProperties": False
        }
    }
}


def _guess_intent(search_info: Any) -> str:
    """根据 search_info 估算检索意图"""
    if not isinstance(search_info, str):
        return "content"
    stripped = search_info.strip()
    if not stripped:
        return "content"
    if stripped == "目录":
        return "catalog"
    for prefix in ("章节标题检索：", "章节标题检索:"):
        if stripped.startswith(prefix):
            return "title"
    return "content"


def _build_wrapper_error(intent: str, message: str) -> Dict[str, Any]:
    """根据意图构造封装层的错误响应"""
    if intent in {"title", "catalog"}:
        return {
            "success": False,
            "source_file": None,
            "content": None,
            "start_page": None,
            "end_page": None,
            "start_chunk_id": None,
            "end_chunk_id": None,
            "error": message
        }
    return {
        "success": False,
        "source_file": None,
        "error": message,
        "retrieved_count": 0,
        "retrieved_summary": None,
        "results": []
    }



def _get_tool_instance() -> ProspectusSearchTool:
    """获取单例工具实例"""
    global _TOOL_INSTANCE
    if _TOOL_INSTANCE is None:
        _TOOL_INSTANCE = ProspectusSearchTool()
    return _TOOL_INSTANCE


def _parse_optional_int(value: Any, label: str) -> Optional[int]:
    """将可选参数转换为整数"""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 的值无效，期望为整数: {value}") from exc


def _parse_bool(value: Any) -> bool:
    """解析布尔型参数，兼容字符串写法"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "t", "y", "真", "是"}:
            return True
        if lowered in {"0", "false", "no", "f", "n", "假", "否"}:
            return False
    raise ValueError(f"布尔参数取值无效: {value}")


def _parse_non_negative_int(value: Any, label: str, default: int = 0) -> int:
    """解析非负整数，并提供默认值"""
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 的值无效，期望为非负整数: {value}") from exc
    if parsed < 0:
        raise ValueError(f"{label} 不能小于 0")
    return parsed


def _normalize_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """整理 LLM 传入的原始参数，转换为工具所需格式"""
    normalized: Dict[str, Any] = {}

    if "fund_code" not in arguments or "search_info" not in arguments:
        raise ValueError("缺少必要字段 fund_code 或 search_info")

    normalized["fund_code"] = str(arguments["fund_code"]).strip()
    normalized["search_info"] = (
        str(arguments["search_info"]) if arguments["search_info"] is not None else ""
    )

    is_expansion = arguments.get("is_expansion", False)
    normalized["is_expansion"] = (
        _parse_bool(is_expansion) if not isinstance(is_expansion, bool) else is_expansion
    )

    normalized["start_page"] = _parse_optional_int(arguments.get("start_page"), "start_page")
    normalized["end_page"] = _parse_optional_int(arguments.get("end_page"), "end_page")
    normalized["start_chunk_id"] = _parse_optional_int(
        arguments.get("start_chunk_id"), "start_chunk_id"
    )
    normalized["end_chunk_id"] = _parse_optional_int(arguments.get("end_chunk_id"), "end_chunk_id")

    normalized["expand_before"] = _parse_non_negative_int(
        arguments.get("expand_before"), "expand_before", default=0
    )
    normalized["expand_after"] = _parse_non_negative_int(
        arguments.get("expand_after"), "expand_after", default=0
    )

    return normalized


def call_prospectus_search(arguments: Dict[str, Any], *, return_json: bool = True) -> Any:
    """执行招募说明书检索，并按需返回 JSON 字符串"""
    intent = _guess_intent(arguments.get("search_info"))

    try:
        params = _normalize_arguments(arguments)
    except Exception as exc:  # noqa: BLE001 直接返回参数解析错误
        result = _build_wrapper_error(intent, f"invalid_arguments: {exc}")
    else:
        try:
            tool = _get_tool_instance()
            result = tool.search_prospectus(**params)
        except Exception as exc:  # noqa: BLE001 保持原始异常信息，便于排查
            result = _build_wrapper_error(intent, f"tool_execution_error: {exc}")

    if return_json:
        return json.dumps(result, ensure_ascii=False)
    return result


def shutdown_tool() -> None:
    """释放缓存的工具实例及相关资源"""
    global _TOOL_INSTANCE
    if _TOOL_INSTANCE is not None:
        _TOOL_INSTANCE.close_connections()
        _TOOL_INSTANCE = None
