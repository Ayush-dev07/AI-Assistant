from tools.base import ToolRegistry
from tools.code.executor import CodeExecutorTool

# Auto-register on import
ToolRegistry.register(CodeExecutorTool())

__all__ = ["CodeExecutorTool"]