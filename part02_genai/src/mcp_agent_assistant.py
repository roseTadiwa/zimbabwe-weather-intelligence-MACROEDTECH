
import asyncio
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MCP_SERVER_FILE = (
    PROJECT_ROOT / "part02_genai" / "src" / "weather_mcp_server.py"
)

SYSTEM_PROMPT = """
You are the Zimbabwe Weather Research Assistant.

You can answer weather research questions and use MCP tools.

Rules:
1. Use the available MCP tools when appropriate.
2. Use get_available_cities to check supported cities.
3. Use get_weather_api_health to check API availability when needed.
4. The predict_temperature tool requires a city, a date and time in
   YYYY-MM-DD HH:MM format, and actual weather measurements.
5. Never invent or guess weather measurements.
6. If measurements are missing, ask the user for them before calling
   the prediction tool.
7. The prediction is for one hour ahead and is not a live observation.
8. Explain prediction results accurately.
9. If asked a general research question, answer clearly without
   making an unnecessary prediction.
"""


async def main():
    print("Connecting to the Zimbabwe Weather MCP Server...")

    client = MultiServerMCPClient(
        {
            "zimbabwe_weather": {
                "transport": "stdio",
                "command": "python",
                "args": [str(MCP_SERVER_FILE)],
                "cwd": str(PROJECT_ROOT),
            }
        }
    )

    tools = await client.get_tools()

    print("\nMCP tools loaded:")
    for tool in tools:
        print(f"- {tool.name}")

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    print("\n" + "=" * 55)
    print("ZIMBABWE WEATHER RESEARCH ASSISTANT — MCP")
    print("Type 'exit' to end.")
    print("=" * 55)

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Assistant: Goodbye!")
                break

            if not question:
                continue

            print("\nAgent is thinking...")

            result = await agent.ainvoke(
                {
                    "messages": [
                        {"role": "user", "content": question}
                    ]
                }
            )

            for message in reversed(result["messages"]):
                if getattr(message, "type", "") == "ai":
                    if message.content:
                        print(f"\nAssistant: {message.content}")
                        break

        except KeyboardInterrupt:
            print("\nAssistant: Session stopped.")
            break

        except Exception as error:
            print(f"\nAn error occurred: {error}")


if __name__ == "__main__":
    asyncio.run(main())
