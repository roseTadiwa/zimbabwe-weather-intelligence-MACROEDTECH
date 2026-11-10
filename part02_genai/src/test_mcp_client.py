
import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    project_root = Path(__file__).resolve().parents[2]
    server_file = project_root / "part02_genai" / "src" / "weather_mcp_server.py"

    server_parameters = StdioServerParameters(
        command="python",
        args=[str(server_file)],
        cwd=str(project_root),
    )

    print("Connecting to the Weather MCP Server...")

    async with stdio_client(server_parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            print("\nConnected successfully!")

            result = await session.list_tools()

            print("\nAvailable MCP tools:")
            for tool in result.tools:
                print(f"- {tool.name}: {tool.description}")

            print("\nTesting the available cities tool...")
            response = await session.call_tool("get_available_cities", {})

            for item in response.content:
                if hasattr(item, "text"):
                    print(item.text)

    print("\nMCP client test completed.")


if __name__ == "__main__":
    asyncio.run(main())
