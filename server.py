
import asyncio
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from openai import AsyncOpenAI


# Инициализируем OpenAI клиента
client = AsyncOpenAI()

# Создаём low-level MCP сервер
server = Server("example-mcp-server")


# -------- TOOLS --------
@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Список доступных инструментов."""
    return [
        types.Tool(
            name="ask_gpt",
            description="Отправляет текст в модель GPT и возвращает ответ.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Текст запроса к модели",
                    }
                },
                "required": ["query"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Обработка вызова инструмента."""
    if name == "ask_gpt":
        query = str(arguments.get("query", ""))

        completion = await client.responses.create(
            model="gpt-4.1-mini",
            input=query,
        )

        return [
            types.TextContent(
                type="text",
                text=completion.output_text,
            )
        ]

    return [
        types.TextContent(
            type="text",
            text=f"Неизвестный инструмент: {name}",
        )
    ]


# -------- PROMPTS (опционально, можно убрать если не нужно) --------
@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="hello",
            description="Пример MCP-промпта",
            arguments=[],
        )
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None = None) -> types.GetPromptResult:
    if name != "hello":
        return types.GetPromptResult(
            description="Промпт не найден",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text="Промпт не найден."),
                )
            ],
        )

    return types.GetPromptResult(
        description="Приветственный промпт",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text="Привет! Это пример MCP-промпта.",
                ),
            )
        ],
    )


# -------- ЗАПУСК СЕРВЕРА ЧЕРЕЗ STDIO --------
async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="example-mcp-server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
