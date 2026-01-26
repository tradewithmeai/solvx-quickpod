"""Tool registry for agent tool execution."""


class ToolRegistry:
    """Registry for tool functions."""

    def __init__(self):
        self._tools = {}

    def register(self, name: str, func):
        """Register a tool function by name."""
        self._tools[name] = func

    def run(self, name: str, args: dict) -> str:
        """Execute a tool by name with given args. Returns result string."""
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name](**args)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools


# Global registry instance
registry = ToolRegistry()
