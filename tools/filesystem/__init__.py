from tools.base import ToolRegistry
from tools.filesystem.file_tool import FileTool

ToolRegistry.register(FileTool())

__all__ = ["FileTool"]