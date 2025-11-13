import os
from typing import Any, Dict, List, Optional

import httpx
from fastmcp import FastMCP
from openai import OpenAI

# ---------- Конфигурация ----------

MCP_SERVER_NAME = "qdrant_rag"

# Переменные окружения
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")  # например: https://your-qdrant-url
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # API key/tокен
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Инициализация клиентов
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
mcp = FastMCP(MCP_SERVER_NAME)


# ---------- Вспомогательные функции ----------

class ConfigError(Exception):
    """Ошибка конфигурации MCP-сервера (нет env переменных и т.п.)."""


def ensure_config() -> None:
    """Проверяем, что все нужные переменные окружения выставлены."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not QDRANT_URL:
        missing.append("QDRANT_URL")
    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")

    if missing:
        raise ConfigError(
            f"Missing env vars: {', '.join(missing)}. "
            "Set them in Render.com → Environment."
        )


async def embed_text(text: str) -> List[float]:
    """Делаем эмбеддинг текста через OpenAI Embeddings API."""
    if not openai_client:
        raise ConfigError("OPENAI_API_KEY is not set")

    resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


async def qdrant_upsert(
    doc_id: str,
    vector: List[float],
    payload: Dict[str, Any],
) -> None:
    """Upsert точки в Qdrant."""
    ensure_config()

    headers = {
        "Content-Type": "application/json",
        "Api-Key": QDRANT_API_KEY,
    }

    body = {
        "points": [
            {
                "id": doc_id,
                "vector": vector,
                "payload": payload,
            }
        ]
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points"
        resp = await client.put(url, headers=headers, json=body)
        resp.raise_for_status()


async def qdrant_search(
    query_vector: List[float],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Поиск по Qdrant по вектору."""
    ensure_config()

    headers = {
        "Content-Type": "application/json",
        "Api-Key": QDRANT_API_KEY,
    }

    body = {
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True,
        "with_vectors": False,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search"
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", [])


# ---------- MCP tools ----------

@mcp.tool
async def ping() -> str:
    """
    Простой healthcheck MCP-сервера.

    Возвращает строку "pong" если сервер жив.
    """
    return "pong"


@mcp.tool
async def store_document(
    doc_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Сохранить текст документа в Qdrant с эмбеддингом.

    Args:
        doc_id: Уникальный ID документа (строка).
        text: Содержимое документа.
        metadata: Дополнительные метаданные (произвольный словарь).

    Returns:
        Строковое сообщение об успешной операции.
    """
    try:
        ensure_config()
        vector = await embed_text(text)
        payload = {"text": text}
        if metadata:
            payload["metadata"] = metadata

        await qdrant_upsert(doc_id, vector, payload)
        return f"Document {doc_id} stored successfully in collection '{QDRANT_COLLECTION}'."
    except Exception as e:
        return f"Error in store_document: {e!r}"


@mcp.tool
async def search_documents(
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Поиск по документам в Qdrant.

    Args:
        query: Текст запроса пользователя.
        top_k: Количество результатов (по умолчанию 5).

    Returns:
        Список найденных документов с полями:
        - id
        - score
        - text
        - metadata (если есть)
    """
    try:
        ensure_config()
        query_vector = await embed_text(query)
        hits = await qdrant_search(query_vector, top_k=top_k)

        results: List[Dict[str, Any]] = []
        for hit in hits:
            payload = hit.get("payload") or {}
            results.append(
                {
                    "id": hit.get("id"),
                    "score": hit.get("score"),
                    "text": payload.get("text"),
                    "metadata": payload.get("metadata"),
                }
            )
        return results
    except Exception as e:
        # Ошибку отдаем как "один элемент списка" — LLM сам поймёт, что что-то пошло не так
        return [{"error": repr(e)}]


# ---------- Точка входа ----------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    # HTTP MCP endpoint будет доступен по адресу: http://0.0.0.0:PORT/mcp
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=port,
        path="/mcp",
    )
