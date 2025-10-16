"""
基础设施公募REITs招募说明书智能检索工具演示脚本
使用智谱GLM-4.6原生Function Calling功能实现

本脚本展示如何使用智谱GLM-4.6模型的原生function calling功能，
通过多轮对话实现对招募说明书的智能检索。
"""
from __future__ import annotations

import argparse
import json
import logging
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
DEFAULT_QA_FILE = Path(__file__).resolve().parent / "招募说明书_qa.json"
LOGGER_NAME = "prospectus_tool_test"
DEFAULT_TEST_QUESTION = "508078.SH基础设施项目是否存在关联交易，如果存在请说明情况。"


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
    parser = argparse.ArgumentParser(description="演示智谱GLM-4.6原生Function Calling调用招募说明书检索工具")
    parser.add_argument(
        "--question",
        default=None,
        help=(
            "用户问题；若不提供则使用脚本内 DEFAULT_TEST_QUESTION，"
            "可直接修改脚本常量实现手动测试"
        ),
    )
    parser.add_argument("--is-expansion", action="store_true", help="是否检索扩募版招募说明书")
    parser.add_argument("--provider", default="zhipu", help="MODEL_CONFIG 中的提供商键，默认 zhipu")
    parser.add_argument("--model", default="glm-4.6", help="MODEL_CONFIG 中的模型键，默认 glm-4.")
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
    parser.add_argument(
        "--skip-thinking",
        action="store_true",
        help="关闭模型思维链（thinking）输出",
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


def _chat_with_tools(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, Any]],
    tool_registry: Dict[str, Any],
    logger: logging.Logger,
    *,
    provider: str,
    max_rounds: int = 20,
    enable_thinking: bool = True,
) -> Tuple[str, bool]:
    """驱动 LLM 多轮对话与工具调用，返回最终回复与是否调用过工具"""
    tools = [PROSPECTUS_SEARCH_TOOL_SPEC]
    tool_used = False
    final_reply = ""

    extra_body_payload = None
    
    if enable_thinking:
        # 只支持 ZhiPu GLM 模型的 thinking 功能
        extra_body_payload = {"thinking": {"type": "enabled"}}

    for round_index in range(1, max_rounds + 1):
        logger.info("=== 第 %d 轮模型请求 ===", round_index)
        logger.debug("发送消息: %s", json.dumps(messages, ensure_ascii=False))

        request_kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        
        # 处理 ZhiPu GLM 模型的推理参数
        if extra_body_payload:
            request_kwargs["extra_body"] = extra_body_payload

        # 记录请求参数用于调试
        #logger.debug("完整请求参数: %s", {k: v for k, v in request_kwargs.items() if k != "messages"})

        response = client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]
        message = choice.message

        # 添加完整响应调试，检查是否有推理信息在其他位置
        #logger.debug("完整响应对象属性: %s", [attr for attr in dir(response) if not attr.startswith('_')])
        #logger.debug("choice 对象属性: %s", [attr for attr in dir(choice) if not attr.startswith('_')])
        
        # 检查是否有推理信息在 choice 级别
        for attr_name in ['reasoning_content', 'reasoning', 'thoughts', 'thinking']:
            if hasattr(choice, attr_name):
                attr_value = getattr(choice, attr_name)
                logger.debug("choice 中找到属性 %s: %s (类型: %s)", attr_name, attr_value, type(attr_value))
        
        # 检查是否有推理信息在 response 级别
        for attr_name in ['reasoning_content', 'reasoning', 'thoughts', 'thinking']:
            if hasattr(response, attr_name):
                attr_value = getattr(response, attr_name)
                logger.debug("response 中找到属性 %s: %s (类型: %s)", attr_name, attr_value, type(attr_value))
        
        # 尝试检查原始 JSON 响应中是否有推理信息
        try:
            raw_response = response.model_dump()
            #logger.debug("原始响应 JSON 键: %s", list(raw_response.keys()))
            
            # 检查是否有推理相关的键
            reasoning_keys = [k for k in raw_response.keys() if 'reason' in k.lower() or 'think' in k.lower()]
            if reasoning_keys:
                logger.debug("找到可能的推理键: %s", reasoning_keys)
                for key in reasoning_keys:
                    logger.debug("推理键 %s 的值: %s", key, raw_response[key])
        except Exception as e:
            logger.debug("获取原始响应 JSON 失败: %s", e)

        reasoning_chunks = _extract_reasoning_chunks_enhanced(message, choice, response)
        if reasoning_chunks:
            logger.info("模型思考过程: %s", "\n".join(reasoning_chunks))
        else:
            # 添加调试信息，帮助诊断推理内容提取问题
            # logger.debug("未找到推理内容，消息属性: %s", 
            #             [attr for attr in dir(message) if not attr.startswith('_')])
            
            # 详细检查消息对象的所有可能包含推理内容的属性
            for attr_name in ['reasoning_content', 'reasoning', 'thoughts', 'thinking']:
                if hasattr(message, attr_name):
                    attr_value = getattr(message, attr_name)
                    logger.debug("找到属性 %s: %s (类型: %s)", attr_name, attr_value, type(attr_value))
            
            # 打印原始消息对象，看看是否有我们遗漏的信息
            logger.debug("完整消息对象: %s", message)
            
            if hasattr(message, 'content'):
                logger.debug("消息内容类型: %s", type(message.content))
                logger.debug("消息内容: %s", message.content)
                if isinstance(message.content, list):
                    logger.debug("内容项目类型: %s", [type(item).__name__ + str(item.get('type', '')) if isinstance(item, dict) else type(item).__name__ for item in message.content])

        assistant_message: Dict[str, Any] = {
            "role": message.role,
            "content": _sanitize_assistant_content(message.content),
        }
        if message.tool_calls:
            assistant_message["tool_calls"] = []
            for call in message.tool_calls:
                assistant_message["tool_calls"].append(
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                )
                logger.info(
                    "模型请求调用工具 %s，参数: %s",
                    call.function.name,
                    call.function.arguments,
                )
        messages.append(assistant_message)

        if not getattr(message, "tool_calls", None):
            final_reply = _stringify_content(message.content)
            logger.info("模型最终回复: %s", final_reply)
            break

        for call in message.tool_calls:
            arguments_str = call.function.arguments or "{}"
            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError as exc:
                logger.error("解析工具参数失败: %s", exc)
                arguments = {}

            if call.function.name not in tool_registry:
                logger.error("收到未知工具调用: %s", call.function.name)
                tool_result = json.dumps(
                    {"success": False, "error": f"unknown_tool: {call.function.name}"},
                    ensure_ascii=False,
                )
            else:
                tool_callable = tool_registry[call.function.name]
                tool_result = tool_callable(arguments)
                tool_used = True

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_result,
                }
            )

        finish_reason = getattr(choice, "finish_reason", "")
        logger.info("本轮 finish_reason: %s", finish_reason)
    else:
        logger.warning("达到最大轮数 %d，结束对话", max_rounds)

    if not final_reply:
        logger.warning("模型在规定轮数内未给出最终回复")

    if not tool_used:
        logger.warning("模型未调用任何工具，请检查提示词或模型配置")

    return final_reply, tool_used


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

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tool_registry = {
        TOOL_NAME: lambda tool_args: _invoke_tool_with_logging(tool_args, logger),
    }

    try:
        final_reply, tool_used = _chat_with_tools(
            client,
            model_cfg["model"],
            messages,
            tool_registry,
            logger,
            provider=args.provider,
            max_rounds=args.max_rounds,
            enable_thinking=not args.skip_thinking,
        )
        logger.info("最终是否调用过工具: %s", tool_used)
        if final_reply:
            logger.info("最终回答:\n%s", final_reply)
        else:
            logger.info("最终回答为空，详见日志分析")
        logger.info("日志文件保存在: %s", log_path)
    except Exception as exc:  # noqa: BLE001 记录所有异常
        logger.exception("执行过程中出现异常: %s", exc)
    finally:
        shutdown_tool()
        logger.info("已关闭工具相关连接")


if __name__ == "__main__":
    main()
