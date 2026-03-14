from tools.base import ToolRegistry, build_tool_definitions_for_llm


def register_all_tools() -> None:
    import tools.web

__all__ = [
    "ToolRegistry",
    "build_tool_definitions_for_llm",
    "register_all_tools",
]