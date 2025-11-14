import asyncio
from mcp.server import Server
from mcp.types import (
    ListToolsResult,
    Tool,
    CallToolResult,
    ListPromptsResult,
    Prompt,
)
from openai import AsyncOpenAI

client = AsyncOpenAI()

server = Server("example-mcp-server")


@server.list_tools()
async def list_tools() -> ListToolsResult:
    return ListToolsResult(
        tools=[
            Tool(
                name="ask_gpt",
                description="Отправляет текст в модель GPT и возвращает ответ.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                },
            )
        ]
    )


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    if name == "ask_gpt":
        query = arguments.get("query", "")

        completion = await client.responses.create(
            model="gpt-4.1-mini",
            input=query,
        )

        return CallToolResult(
            content=[{"type": "text", "text": completion.output_text}]
        )

    return CallToolResult(
        content=[{"type": "text", "text": f"Неизвестный инструмент: {name}"}]
    )


@server.list_prompts()
async def list_prompts() -> ListPromptsResult:
    return ListPromptsResult(
        prompts=[
            Prompt(
                name="hello",
                description="Пример MCP-промпта",
                arguments=[],
            )
        ]
    )


@server.get_prompt()
async def get_prompt(name: str):
    if name == "hello":
        return {
            "content": [
                {"type": "text", "text": "Привет! Это пример MCP промпта."}
            ]
        }

    return {
        "content": [
            {"type": "text", "text": "Промпт не найден."}
        ]
    }


async def main():
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
