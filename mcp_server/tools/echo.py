from mcp.server.fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    def shout(text: str) -> str:
        """Uppercase the given text."""
        return text.upper()