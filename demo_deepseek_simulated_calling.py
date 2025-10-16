"""
基础设施公募REITs招募说明书智能检索工具演示脚本
使用DeepSeek Reasoner模拟Function Calling功能实现

本脚本展示如何利用DeepSeek Reasoner模型的强大推理能力，
通过模拟function calling的方式实现对招募说明书的智能检索。
相比原生function calling，这种方式能更好地利用模型的推理能力。
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from model_config import MODEL_CONFIG
from intelligent_search.tool_entry import (
    PROSPECTUS_SEARCH_TOOL_SPEC,
    TOOL_NAME,
    call_prospectus_search,
    shutdown_tool,
)

LOG_DIR = Path(__file__).resolve().parent / "log"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_QA_FILE = Path(__file__).resolve().parent / "招募说明书_qa.json"
LOGGER_NAME = "prospectus_tool_test"
DEFAULT_TEST_QUESTION = "根据180101.SZ招募说明书中的基础设施基金整体架构章节内容，该基金是否有外部借款（外部杠杆）安排，如果有请详细介绍。"


def setup_logging() -> Tuple[logging.Logger, Path]:
    """初始化日志，输出到控制台与文件"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"prospectus_tool_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("日志写入路径: %s", log_path)
    return logger, log_path


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="演示DeepSeek Reasoner模拟Function Calling调用招募说明书检索工具")
    parser.add_argument(
        "--question",
        default=None,
        help=(
            "用户问题；若不提供则使用脚本内 DEFAULT_TEST_QUESTION，"
            "可直接修改脚本常量实现手动测试"
        ),
    )
    parser.add_argument("--is-expansion", action="store_true", help="是否检索扩募版招募说明书")
    parser.add_argument("--provider", default="deepseek", help="MODEL_CONFIG 中的提供商键，默认 deepseek")
    parser.add_argument("--model", default="deepseek-reasoner", help="MODEL_CONFIG 中的模型键，默认 deepseek-reasoner")
    parser.add_argument(
        "--qa-file",
        type=Path,
        default=DEFAULT_QA_FILE,
        help="描述招募说明书章节内容的 QA JSON 文件路径",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=20,
        help="LLM 交互的最大轮数，默认 20 轮",
    )
    return parser.parse_args()


def _extract_model_config(provider: str, model_name: str) -> Dict[str, str]:
    try:
        return MODEL_CONFIG[provider][model_name]
    except KeyError as exc:
        raise SystemExit(f"未找到模型配置 {provider}.{model_name}: {exc}")


def load_reference_qas(path: Path, logger: logging.Logger) -> List[Dict[str, str]]:
    """读取招募说明书结构参考问答，若不存在则返回空列表"""
    if not path.exists():
        logger.warning("未找到参考 QA 文件: %s", path)
        return []

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            logger.info("载入参考 QA 条目 %d 条", len(data))
            return [item for item in data if isinstance(item, dict)]
        logger.warning("参考 QA 文件格式异常，期望列表，实际类型: %s", type(data))
    except json.JSONDecodeError as exc:
        logger.error("解析参考 QA 文件失败: %s", exc)
    except Exception as exc:  # noqa: BLE001 捕获所有异常记录日志
        logger.error("读取参考 QA 文件时出错: %s", exc)
    return []


def format_reference_text(qas: List[Dict[str, str]], limit: int = 20) -> str:
    """将参考 QA 转换为系统提示中的文本"""
    if not qas:
        return "（未提供参考问答）"

    lines: List[str] = []
    for idx, item in enumerate(qas, start=1):
        question = item.get("q", "-").strip()
        answer = item.get("a", "-").strip()
        lines.append(f"{idx}. 问题：{question}\n   要点：{answer}")
        if limit and idx >= limit:
            if len(qas) > limit:
                lines.append(f"…… 其余 {len(qas) - limit} 条问答已省略")
            break
    return "\n".join(lines)


def build_system_prompt(reference_text: str) -> str:
    """生成系统提示词，指导 LLM 的工作流程和工具使用方式"""
    return (
        "你是一名熟悉中国基础设施公募REITs招募说明书结构的专业助手，全部的基础设施公募REITs的招募说明书文本已被按照200-1500字符切分成众多的文本块，你的目标是在回答前通过多轮工具调用获取相应的原文信息，确保依据充分。请严格执行以下作业流程：\n"
        "一、准备阶段\n"
        "1. 阅读下方提供的招募说明书参考章节要点，了解各章节常见内容与排布顺序。\n"
        "2. 仔细研读用户问题，主动识别其中的基金代码以及适用的招募说明书版本（首发/扩募），如未明确则视为首发；如无法确定基金代码，应先向用户确认后再继续。\n"
        "3. 第一轮工具调用必须获取该基金的招募说明书目录，调用参数需包含识别出的 fund_code 及必要的 is_expansion（如需） 标记，search_info=\"目录\"。\n"
        "二、定位具体章节阶段\n"
        "4. 对照目录和参考章节要点，推断问题所属章节。示例：若问题是“基础设施项目最近三年及一期的EBITDA是多少？”，参考章节要点说明【基础设施项目基本情况】章节包含历史收益信息，其中很可能包括EBITDA指标信息，则应检索该章节。\n"
        "5. 定位目标章节正文：有两种方法：\n"
        "（1）使用start_page等于目录中目标章节的页码，search_info为空，并且合理确定end_page，end_page确定方式是：如果目录中下一章节的页码与目标章节页码的差值在100以内，则可以一次性获取该章节全部内容，end_page选取略大于下一个章节的页码的数即可（因为实际的页码可能会大于目录中的页码，但是偏差不超过30页）；如果目录中下一章节的页码与目标章节页码的差值超过100则说明该章节内容过长，则end_page适当选取（比如比start_page大100），先预览一部分信息，后续再逐步往后扩展。\n"
        "（2）如果目录没有页码或前一种方法失败，则使用“章节标题检索：目标章节标题”定位章节首段，例如search_info=“第十四部分 基础设施项目基本情况”， expand_after设置一个数（例如5），预览后续内容（这种方法有可能会返回的内容仅是对于目标章节的引用而不是目标章节的正文）。\n两种方式都需要记录返回的页码与 chunk_id。\n"
        "三、章节深入检索阶段（如需）\n"
        "6. 若首轮未提取完整章节文本、且未覆盖目标信息，则继续往后扩展文本块以提取该章节剩余部分，可参考参考章节要点中对于该章节内容结构介绍以及页数范围，决定是否进一步往后扩展以及往后扩展多少页或多少文本块。可按页码或 chunk_id 连续提取，例如上一轮结束 chunk_id=100，则下一轮设置 start_chunk_id=101、end_chunk_id=120、search_info=空字符串；也可按页码区间提取。章节内容过长的可不断重复此操作，直至获得答案或已获得该章节完整信息。\n"
        "四、调整范围（如需）\n"
        "7. 若已获取目标章节完整信息，但仍未获得答案，则需要更换检索章节，可考虑以下两种情况：\n"
        "（1）若判断其他章节可能有答案，则找出最可能的章节重复上述步骤；\n"
        "（2）如无法准确判断目标章节，则可以考虑使用search_info=“内容检索：关键词”获取包含关键词文本块，例如直接检索问题的关键词或者检索参考章节要点中提到的可能出现的小标题，结合适当的expand_before/expand_after（比如均为1），工具将返回多条包含检索关键词的文本信息（最多20条）及对应的页码和chunk_id，然后从中找出最有可能含有答案的文本信息，并根据其chunk_id或页码进一步扩展以获得完整信息。如果知道大致范围，可结合已知页码或已知chunk_id的使用 start_chunk_id/end_chunk_id 或 start_page/end_page 限定范围进行检索，但无法判断检索范围，可不限制范围的使用search_info=“内容检索：关键词”，则会在全文内检索。\n"
        "四、回答阶段\n"
        "8. 汇总结果时务必引用工具返回文本中的证据，并标注来源（如所在具体章节标题、表格标题等（如有），无需提供页码范围）；若信息不足，需要说明缺口及下一步建议。\n"
        "五、工具说明与注意事项\n"
        "- fund_code：必填，请使用从用户问题中解析出的基金代码。\n"
        "- search_info：必填，支持：\n"
        "  • “目录”——获取完整目录。请注意，目录中的页码并非实际页码，实际页码可能会大于目录中的页码，但是偏差一般在30页以内；\n"
        "  • “章节标题检索：目标章节标题”——定位章节开头，请注意，请提供目录中准确的标题信息及序号，例如：第十四部分 基础设施项目基本情况”；\n"
        "  • “内容检索：需要检索的内容”——检索关键信息，例如检索问题的关键词或者检索参考章节要点中提到的可能出现的小标题关键词。\n"
        "  • 空字符串——直接返回限定范围内的原文文本块。\n"
        "- is_expansion：选填，为 True 时检索扩募版招募说明书。\n"
        "- start_page/end_page、start_chunk_id/end_chunk_id：选填，限制检索范围。\n"
        "- expand_before/expand_after：选填，控制返回的上下文扩展文本块数量，检索结果文本块仅为单个文本块，但是使用这两个参数可以将检索结果文本块前后的文本块一同返回，以获取完整的上下文；单个文本块约 200-1500 字，请结合需求设定，初步预览可考虑前后各扩展1个文本块，找到需要的文本信息后可考虑上下文多个文本块。\n"
        "- 工具调用非常灵活，可以多轮调用，目标是获取需要的文本信息，情形包括但不限于：1）获取招募说明书目录：search_info填写“目录”；2）获取目录展示的特定章节标题所在的正本文本块，根据目录中的页码合理填写start_page和end_page，或者search_info填写“章节标题检索：目标章节标题”，expand_after根据需要填写；3）检索特定信息，search_info填写“内容检索：需要检索的内容”，根据已知的信息确定是否需要填写start_page/end_page/start_chunk_id/end_chunk_id。4）获取特定页面范围内/特定chunk_id范围内的文本信息，这时search_info填写为空，根据需要填写start_page和end_page（或start_chunk_id和end_chunk_id）。\n"
        "- 在目标章节页数可控情况下（100页以内），优先探索完毕完整的章节文本信息，除非目标章节内容特别长或者确实没有答案，才考虑使用内容检索功能。在目标章节页数比较长时，已获得的该章节信息不够时，仍需对该章节进行探索时，可以使用内容检索，但尽量结合已获得信息的chunk_id或页码缩小检索窗口的范围。\n"
        "- 工具返回出内容：将返回执行状态（success）、文件名称（source_file）、检索结果的数量（retrieved_count）、每个检索结果对应的文本信息（扩展后）（text或content）、每个文本信息对应的起始页码（start_page）、每个文本信息对应的终止页码（end_page）、每个文本信息对应的起始chunk_id（start_chunk_id）、每个文本信息对应的起始chunk_id（end_chunk_id）。\n"        
        "六、特殊提醒\n"
        "- 章节标题可能与参考章节要点中存在措辞差异，匹配时需灵活处理。\n"
        "- 参考章节要点提供的是通用结构，与真实招募说明书的内容顺序高度相似，可据此推断但不得武断。请多多结合参考章节要点锁定范围。\n"
        "- 若模型出现遗漏工具调用、未按目录操作等情况，需主动纠正并重新按流程执行。\n"
        "- 始终以中文作答，禁止凭空推测和编造数据。\n"
        "- 调用工具时，每一轮请只调用一次。\n"
        "\n参考章节要点：\n"
        f"{reference_text}\n"
        "如参考信息不足，可在作答中说明需要补充的材料。"
    )


def build_user_prompt(question: str, is_expansion: bool) -> str:
    """直接返回用户原始问题"""
    _ = is_expansion  # 保留参数以便脚本接口兼容
    return question


def build_deepseek_reasoner_enhanced_prompt(original_system_prompt: str, tools_schema: List[Dict[str, Any]]) -> str:
    """为 DeepSeek Reasoner 构建增强的系统提示词，在原有基础上追加格式指导"""
    
    # 将 tools schema 转为 JSON 字符串用于展示
    tools_json = json.dumps(tools_schema, ensure_ascii=False, indent=2)
    
    reasoner_addition = f"""

## DeepSeek Reasoner 专用指导

### 重要说明
你正在模拟多轮工具调用对话。由于技术限制，你无法直接调用工具，但可以表达调用指令。每轮对话后，我会根据你的指令执行工具调用并返回结果。

### 输出格式要求
每轮回复必须按照以下格式输出：

#### 格式A：需要调用工具
```
TOOL_CALL:
{{
    "fund_code": "基金代码",
    "search_info": "检索信息",
    "is_expansion": false,
    "start_page": null,
    "end_page": null,
    "start_chunk_id": null,
    "end_chunk_id": null,
    "expand_before": 0,
    "expand_after": 0
}}
```
请注意，如本轮需要调用工具则只返回TOOL_CALL，并且只填写本轮需要的参数，未使用的字段不要出现，也不要为了凑格式填写 null 或默认值。

#### 格式B：给出最终答案
```
FINAL_ANSWER:
具体的最终答案内容，并标注来源（如所在具体章节标题、表格标题等（如有），无需提供页码范围）。
```

### 工具调用规则
可用工具的完整规格如下：
{tools_json}

### 参数使用指导
**必填参数：**
- fund_code: 从用户问题中识别的基金代码
- search_info: 根据当前需求选择合适的检索方式

**可选参数（仅在需要时填写非默认值）：**
- is_expansion: 只有检索扩募版时填true
- start_page/end_page: 限定页码范围时使用
- start_chunk_id/end_chunk_id: 连续检索chunk时使用
- expand_before/expand_after: 需要扩展上下文时使用

### 执行要求
- 严格按照上述格式输出，否则系统无法解析
- TOOL_CALL 中仅填写本轮真正需要的参数
- 每轮只调用一次工具
- JSON格式必须正确，不要添加注释

现在开始分析用户问题并逐步检索相关信息："""

    return original_system_prompt + reasoner_addition


def _stringify_content(content: Any) -> str:
    """将 OpenAI 返回的 content 转为字符串"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _extract_reasoning_chunks(message: Any) -> List[str]:
    """从模型返回的消息中提取思考/推理文本"""
    chunks: List[str] = []

    for attr in ("reasoning_content", "reasoning"):
        value = getattr(message, attr, None)
        if value:
            chunks.append(_stringify_content(value))

    content = getattr(message, "content", None)
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"reasoning", "thought", "thinking"}:
                    text = item.get("text")
                    if text:
                        chunks.append(text)

    return [chunk for chunk in chunks if chunk]


def _extract_reasoning_chunks_enhanced(message: Any, choice: Any = None, response: Any = None) -> List[str]:
    """增强版推理内容提取，从消息、choice 和 response 多个层级查找"""
    chunks: List[str] = []

    # 从消息层级提取
    chunks.extend(_extract_reasoning_chunks(message))
    
    # 从 choice 层级提取
    if choice:
        for attr in ("reasoning_content", "reasoning", "thoughts", "thinking"):
            value = getattr(choice, attr, None)
            if value:
                chunks.append(_stringify_content(value))
    
    # 从 response 层级提取
    if response:
        for attr in ("reasoning_content", "reasoning", "thoughts", "thinking"):
            value = getattr(response, attr, None)
            if value:
                chunks.append(_stringify_content(value))

    return [chunk for chunk in chunks if chunk]


def _sanitize_assistant_content(raw_content: Any) -> Any:
    """移除模型返回的推理片段，避免下轮请求报错"""
    if isinstance(raw_content, list):
        filtered: List[Any] = []
        for item in raw_content:
            if isinstance(item, dict) and item.get("type") in {"reasoning", "thought", "thinking"}:
                continue
            filtered.append(item)
        return filtered
    return raw_content



def _parse_reasoner_output(content: str, reasoning_content: str, logger: logging.Logger) -> Dict[str, Any]:
    """解析 DeepSeek Reasoner 的输出内容和推理过程"""
    result = {
        "type": None,  # "tool_call", "final_answer", "format_error", "error"
        "analysis": "",
        "tool_call": None,
        "final_answer": None,
        "reasoning": reasoning_content,
        "errors": []
    }
    
    # 1. 提取分析部分
    analysis_patterns = [
        r'本轮分析[：:]\s*(.*?)(?=TOOL_CALL:|FINAL_ANSWER:|$)',
        r'最终分析[：:]\s*(.*?)(?=FINAL_ANSWER:|$)'
    ]
    
    for pattern in analysis_patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            result["analysis"] = match.group(1).strip()
            break
    
    # 2. 检查工具调用
    tool_call_pattern = r'TOOL_CALL:\s*(\{.*?\})'
    tool_match = re.search(tool_call_pattern, content, re.DOTALL)
    
    if tool_match:
        try:
            tool_call_json = tool_match.group(1)
            logger.debug("解析工具调用 JSON: %s", tool_call_json)
            tool_call = json.loads(tool_call_json)
            
            # 验证必填参数
            required_params = ["fund_code", "search_info"]
            missing_params = [p for p in required_params if p not in tool_call or tool_call[p] is None]
            if missing_params:
                result["errors"].append(f"缺少必填参数: {missing_params}")
                result["type"] = "error"
                logger.warning("工具调用缺少必填参数: %s", missing_params)
                return result
            
            # 保留模型填写的参数原样传递，避免额外填充默认值
            
            result["type"] = "tool_call"
            result["tool_call"] = tool_call
            logger.info("成功解析工具调用: %s", json.dumps(tool_call, ensure_ascii=False))
            return result
            
        except json.JSONDecodeError as e:
            result["errors"].append(f"JSON解析失败: {e}")
            result["type"] = "error"
            logger.error("工具调用 JSON 解析失败: %s, 原始内容: %s", e, tool_match.group(1))
            return result
    
    # 3. 检查最终答案
    final_pattern = r'FINAL_ANSWER:\s*(.*)\s*$'
    final_match = re.search(final_pattern, content, re.DOTALL)
    
    if final_match:
        final_answer = final_match.group(1).strip()
        if not final_answer:
            result["errors"].append("最终答案不能为空")
            result["type"] = "error"
            logger.warning("最终答案为空")
        else:
            result["type"] = "final_answer"
            result["final_answer"] = final_answer
            logger.info("解析到最终答案")
        return result
    
    # 4. 格式错误处理
    result["type"] = "format_error"
    result["errors"].append("未找到有效的TOOL_CALL或FINAL_ANSWER格式")
    logger.warning("输出格式错误，未找到TOOL_CALL或FINAL_ANSWER")
    return result


def _chat_with_deepseek_reasoner(
    client: OpenAI,
    model_name: str,
    user_question: str,
    original_system_prompt: str,
    tool_registry: Dict[str, Any],
    logger: logging.Logger,
    max_rounds: int = 20
) -> Dict[str, Any]:
    """DeepSeek Reasoner 专用的对话流程"""
    
    # 构建增强的系统提示词
    tools_schema = [PROSPECTUS_SEARCH_TOOL_SPEC]
    enhanced_system_prompt = build_deepseek_reasoner_enhanced_prompt(original_system_prompt, tools_schema)
    
    # 初始化对话
    messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": user_question}
    ]
    
    # 结果记录
    conversation_rounds = []
    all_reasoning = []
    tool_calls_made = []
    
    for round_index in range(1, max_rounds + 1):
        logger.info(f"=== 第 {round_index} 轮 DeepSeek Reasoner 对话 ===")
        
        # 调用模型（不传递 tools 参数以获取推理过程）
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
        except Exception as e:
            logger.error(f"第 {round_index} 轮调用模型失败: {e}")
            return {
                "success": False,
                "error": f"模型调用失败: {e}",
                "final_answer": "",
                "conversation_rounds": conversation_rounds,
                "all_reasoning": all_reasoning,
                "tool_calls": tool_calls_made,
                "total_rounds": round_index - 1
            }
        
        choice = response.choices[0]
        message = choice.message
        
        # 提取推理内容和回复内容
        reasoning_content = getattr(message, 'reasoning_content', '') or ''
        content = getattr(message, 'content', '') or ''
        
        if reasoning_content:
            logger.info(f"第 {round_index} 轮推理过程:\n{reasoning_content}")
            all_reasoning.append(f"=== 第 {round_index} 轮推理 ===\n{reasoning_content}")
        
        logger.info(f"第 {round_index} 轮回复:\n{content}")
        
        # 解析输出
        parsed = _parse_reasoner_output(content, reasoning_content, logger)
        
        # 记录本轮对话
        round_data = {
            "round_number": round_index,
            "reasoning_content": reasoning_content,
            "response_content": content,
            "parsed_analysis": parsed["analysis"],
            "parsed_type": parsed["type"],
            "tool_call_params": parsed.get("tool_call"),
            "final_answer": parsed.get("final_answer"),
            "errors": parsed["errors"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        conversation_rounds.append(round_data)
        
        # 添加助手回复到对话历史（只添加 content，不添加 reasoning_content）
        messages.append({"role": "assistant", "content": content})
        
        # 根据解析结果采取行动
        if parsed["type"] == "tool_call":
            # 执行工具调用
            tool_call = parsed["tool_call"]
            logger.info(f"执行工具调用: {json.dumps(tool_call, ensure_ascii=False)}")
            
            try:
                # 执行工具
                tool_response = tool_registry[TOOL_NAME](tool_call)
                tool_call_record = {
                    "round": round_index,
                    "params": tool_call,
                    "result": tool_response,
                    "success": True
                }
                tool_calls_made.append(tool_call_record)
                
                # 更新当前轮记录
                conversation_rounds[-1]["tool_execution_result"] = tool_response
                conversation_rounds[-1]["tool_success"] = True
                
                # 将工具结果返回给模型
                tool_feedback = f"工具调用已执行完成，结果如下：\n\n{tool_response}\n\n请基于此结果继续分析或给出最终答案。记住要按照指定格式输出。"
                messages.append({"role": "user", "content": tool_feedback})
                
                logger.info("工具调用成功，已将结果反馈给模型")
                
            except Exception as e:
                error_msg = f"工具调用失败: {str(e)}"
                logger.error(error_msg)
                
                tool_call_record = {
                    "round": round_index,
                    "params": tool_call,
                    "result": None,
                    "success": False,
                    "error": str(e)
                }
                tool_calls_made.append(tool_call_record)
                
                conversation_rounds[-1]["tool_execution_result"] = error_msg
                conversation_rounds[-1]["tool_success"] = False
                
                retry_feedback = f"{error_msg}\n\n请检查参数并重试，或采用其他检索策略。记住要按照指定格式输出。"
                messages.append({"role": "user", "content": retry_feedback})
        
        elif parsed["type"] == "final_answer":
            # 获得最终答案，结束对话
            logger.info("获得最终答案，对话成功结束")
            return {
                "success": True,
                "final_answer": parsed["final_answer"],
                "conversation_rounds": conversation_rounds,
                "all_reasoning": all_reasoning,
                "tool_calls": tool_calls_made,
                "total_rounds": round_index,
                "complete_reasoning_log": "\n\n".join(all_reasoning)
            }
        
        elif parsed["type"] in ["format_error", "error"]:
            # 格式错误，给予提示
            error_details = "; ".join(parsed["errors"])
            prompt_msg = (
                f"输出格式有误：{error_details}\n\n"
                "请严格按照以下格式输出：\n"
                "1. 如需调用工具，使用格式：\n"
                "TOOL_CALL:\n"
                "{ 仅填写本轮需要的参数 }\n\n"
                "2. 如给出最终答案，使用格式：\n"
                "FINAL_ANSWER:\n"
                "具体答案内容"
            )
            messages.append({"role": "user", "content": prompt_msg})
            logger.warning(f"第 {round_index} 轮格式错误，已给予纠正提示")
    
    # 达到最大轮数
    logger.warning(f"达到最大轮数 {max_rounds}，对话结束")
    return {
        "success": False,
        "error": f"达到最大轮数 {max_rounds}",
        "final_answer": "达到最大对话轮数，未能获得完整答案",
        "conversation_rounds": conversation_rounds,
        "all_reasoning": all_reasoning,
        "tool_calls": tool_calls_made,
        "total_rounds": max_rounds,
        "complete_reasoning_log": "\n\n".join(all_reasoning)
    }


def _invoke_tool_with_logging(arguments: Dict[str, Any], logger: logging.Logger) -> str:
    """调用工具并记录日志，默认返回 JSON 字符串"""
    logger.info(
        "调用工具 %s，入参: %s",
        TOOL_NAME,
        json.dumps(arguments, ensure_ascii=False, sort_keys=True),
    )
    result = call_prospectus_search(arguments, return_json=True)
    logger.info("工具返回: %s", result)
    return result


def main() -> None:
    args = _parse_arguments()
    logger, log_path = setup_logging()

    question = args.question if args.question is not None else DEFAULT_TEST_QUESTION
    if args.question is None:
        logger.info("未通过命令行提供问题，使用 DEFAULT_TEST_QUESTION: %s", question)

    logger.info(
        "启动测试：question=%s, is_expansion=%s, provider=%s, model=%s",
        question,
        args.is_expansion,
        args.provider,
        args.model,
    )

    qas = load_reference_qas(args.qa_file, logger)
    reference_text = format_reference_text(qas)
    system_prompt = build_system_prompt(reference_text)
    user_prompt = build_user_prompt(question, args.is_expansion)

    logger.debug("系统提示词:\n%s", system_prompt)
    logger.info("用户初始消息:\n%s", user_prompt)

    model_cfg = _extract_model_config(args.provider, args.model)
    client = OpenAI(api_key=model_cfg["api_key"], base_url=model_cfg["base_url"])

    tool_registry = {
        TOOL_NAME: lambda tool_args: _invoke_tool_with_logging(tool_args, logger),
    }

    run_started_at = datetime.now().isoformat()
    reasoner_result: Dict[str, Any] | None = None
    error_message: str | None = None

    try:
        provider_lower = args.provider.lower()
        model_lower = model_cfg["model"].lower()

        if provider_lower != "deepseek" or "reasoner" not in model_lower:
            msg = (
                f"当前脚本仅支持 DeepSeek Reasoner 模型，实际配置: provider={args.provider}, "
                f"model={model_cfg['model']}"
            )
            reasoner_result = {
                "success": False,
                "error": msg,
                "final_answer": "",
                "conversation_rounds": [],
                "tool_calls": [],
            }
            logger.error(msg)
            raise SystemExit(msg)

        logger.info("检测到 DeepSeek Reasoner 模型，使用专用对话流程")

        reasoner_result = _chat_with_deepseek_reasoner(
            client,
            model_cfg["model"],
            user_prompt,
            system_prompt,
            tool_registry,
            logger,
            max_rounds=args.max_rounds,
        )
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)
        logger.exception("执行过程中出现异常: %s", exc)
        if reasoner_result is None:
            reasoner_result = {
                "success": False,
                "error": error_message,
                "final_answer": "",
                "conversation_rounds": [],
                "tool_calls": [],
            }
    else:
        tool_calls = reasoner_result.get("tool_calls", [])
        tool_used = len(tool_calls) > 0
        final_reply = reasoner_result.get("final_answer", "") or ""
        success_flag = bool(reasoner_result.get("success"))

        logger.info("=== DeepSeek Reasoner 对话完成 ===")
        logger.info("成功状态: %s", success_flag)
        logger.info("总轮数: %s", reasoner_result.get("total_rounds"))
        logger.info("工具调用次数: %s", len(tool_calls))

        if success_flag:
            if final_reply:
                logger.info("最终回答:\n%s", final_reply)
            else:
                logger.info("模型标记成功，但最终回答为空")
        else:
            logger.warning("对话失败: %s", reasoner_result.get("error", "未知错误"))
            if final_reply:
                logger.info("部分回答:\n%s", final_reply)
            else:
                logger.info("未获取到模型有效回答，详见日志分析")

        logger.info("完整推理日志已记录，详见调试输出")
        logger.info("最终是否调用过工具: %s", tool_used)
        logger.info("日志文件保存在: %s", log_path)
    finally:
        run_finished_at = datetime.now().isoformat()

        rounds_payload: List[Dict[str, Any]] = []
        if reasoner_result:
            for round_entry in reasoner_result.get("conversation_rounds", []):
                tool_params = round_entry.get("tool_call_params")
                tool_call_payload = None
                if tool_params:
                    tool_call_payload = {
                        "name": TOOL_NAME,
                        "arguments": tool_params,
                    }

                tool_exec_result = round_entry.get("tool_execution_result")
                tool_response_payload = None
                if tool_exec_result is not None:
                    tool_response_payload = {
                        "raw": tool_exec_result,
                        "success": round_entry.get("tool_success"),
                    }

                rounds_payload.append(
                    {
                        "round": round_entry.get("round_number"),
                        "timestamp": round_entry.get("timestamp"),
                        "think": round_entry.get("reasoning_content") or None,
                        "assistant_reply": round_entry.get("response_content") or None,
                        "tool_call": tool_call_payload,
                        "tool_response": tool_response_payload,
                    }
                )

        if reasoner_result:
            final_answer = reasoner_result.get("final_answer", "") or ""
            success_flag = bool(reasoner_result.get("success"))
            error_text = reasoner_result.get("error")
        else:
            final_answer = ""
            success_flag = False
            error_text = error_message

        base_dir = Path(__file__).resolve().parent
        log_rel_path = None
        if log_path is not None:
            try:
                log_rel_path = str(log_path.relative_to(base_dir))
            except ValueError:
                log_rel_path = str(log_path)

        output_payload = {
            "question": question,
            "run_id": log_path.stem if log_path else None,
            "run_started_at": run_started_at,
            "run_finished_at": run_finished_at,
            "rounds": rounds_payload,
            "final_answer": final_answer,
            "success": success_flag,
            "error": error_text,
            "log_file": log_rel_path,
        }

        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            if log_path is not None:
                output_filename = f"{log_path.stem}.json"
            else:
                output_filename = f"prospectus_tool_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path = OUTPUT_DIR / output_filename
            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(output_payload, output_file, ensure_ascii=False, indent=2)
            logger.info("会话记录已保存至: %s", output_path)
        except Exception as json_exc:  # noqa: BLE001
            logger.exception("保存会话 JSON 失败: %s", json_exc)

        shutdown_tool()
        logger.info("已关闭工具相关连接")


if __name__ == "__main__":
    main()
