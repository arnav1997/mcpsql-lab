import platform
from datetime import datetime
from mcp.server.fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    def system_info() -> dict:
        """Return simple host information (no PII)."""
        return {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
