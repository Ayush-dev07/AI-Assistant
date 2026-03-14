from tools.base import ToolRegistry
from tools.web.browser_tool import BrowserTool

ToolRegistry.register(BrowserTool())

__all__ = ["BrowserTool"]