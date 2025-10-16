"""
基础设施公募REITs招募说明书智能检索工具包

基于对传统RAG技术改造的创新检索系统，专门设计用于处理复杂的金融公告文档。
模拟人类阅读文件和查找目标信息的流程，特别适用于章节繁多、内容冗长、
格式相对规范的招募说明书等金融公告。

主要特性：
- 六阶段智能检索流程（准备→定位→深入→调整→回答→验证）
- 多模式检索支持（关键词、语义向量、混合检索）
- 智能文本块扩展和范围限制
- 支持原生Function Calling和模拟Function Calling两种LLM交互模式

核心组件：
- prospectus_search_tool: 核心检索工具类
- tool_entry: LLM工具调用入口
- core: 文件管理和目录检索核心功能
- searchers: 多种检索器实现
- utils: 文本处理和工具函数

作者：[您的名字]
版本：1.0.0
"""

from .prospectus_search_tool import ProspectusSearchTool
from .tool_entry import (
    PROSPECTUS_SEARCH_TOOL_SPEC,
    TOOL_NAME,
    call_prospectus_search,
    shutdown_tool
)

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    'ProspectusSearchTool',
    'PROSPECTUS_SEARCH_TOOL_SPEC', 
    'TOOL_NAME',
    'call_prospectus_search',
    'shutdown_tool'
]