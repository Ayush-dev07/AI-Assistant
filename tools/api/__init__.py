from tools.api.rest_tool import APICallerTool
from tools.base import ToolRegistry

# Auto-register the open (all-domains) instance on import
ToolRegistry.register(APICallerTool())

__all__ = ["APICallerTool"]