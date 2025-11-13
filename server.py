import os
from typing import Any, Dict, List, Optional

import httpx
from fastmcp import FastMCP
from openai import OpenAI


# ---------- Конфигурация ----------

MCP_SERVER_NAME = "qdrant_rag"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")            # пример: https://xxx.eu-central-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Инициализация FastMCP
mcp = FastMCP(MCP_SERVER_NAME)


# ---------- Вспомогательные функции ----------

class ConfigError(Exception):
    """Ошибка конфигурации MCP-сервера (нет env-переменных и т.п.)."""
    pass


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
            f"Set them in Render Environment."
        )


def embed_text(text: str) -> List[float]:
    """Синхронно получаем эмбеддинг текста через OpenAI Embeddings API."""
    if not openai_client:
        raise ConfigError("OPENAI_API_KEY is not set")

    resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return resp.data[0].embedding


def qdrant_upsert(
    doc_id: str,
    vector: List[float],
    payload: Dict[str, Any],
) -> None:
    """Создаём/обновляем точку в Qdrant."""
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

    base_url = QDRANT_URL.rstrip("/")

    with httpx.Client(timeout=30.0) as client:
        url = f"{base_url}/collections/{QDRANT_COLLECTION}/points"
        resp = client.put(url, headers=headers, json=body)
        resp.raise_for_status()


def qdrant_search(
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

    base_url = QDRANT_URL.rstrip("/")

    with httpx.Client(timeout=30.0) as client:
        url = f"{base_url}/collections/{QDRANT_COLLECTION}/points/search"
        resp = client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", [])


# ---------- MCP tools ----------

@mcp.tool()
def ping() -> str:
    """
    Healthcheck MCP-сервера.

    Возвращает "pong", если сервер жив и конфиг валиден.
    """
    ensure_config()
    return "pong"


@mcp.tool()
def store_document(
    doc_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Сохранить текст документа в Qdrant с эмбеддингом.

    Args:
        doc_id: Уникальный ID документа.
        text: Содержимое документа.
        metadata: Доп. метаданные (любой JSON-объект).

    Returns:
        Статус-строка об успешной операции или ошибке.
    """
    try:
        ensure_config()
        vector = embed_text(text)
        payload: Dict[str, Any] = {"text": text}
        if metadata:
            payload["metadata"] = metadata

        qdrant_upsert(doc_id, vector, payload)
        return f"Document {doc_id} stored successfully in collection '{QDRANT_COLLECTION}'."
    except Exception as e:
        return f"Error in store_document: {e!r}"


@mcp.tool()
def search_documents(
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Поиск по документам в Qdrant.

    Args:
        query: Текстовый запрос пользователя.
        top_k: Количество результатов (по умолчанию 5).

    Returns:
        Список найденных документов:
        [
          {
            "id": ...,
            "score": ...,
            "text": ...,
            "metadata": {...}
          },
          ...
        ]
        либо один элемент с ключом "error" при ошибке.
    """
    try:
        ensure_config()
        query_vector = embed_text(query)
        hits = qdrant_search(query_vector, top_k=top_k)

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
        return [{"error": repr(e)}]


# ---------- Точка входа ----------
if __name__ == "__main__":
    # Render пробрасывает порт в переменную PORT
    port = int(os.environ.get("PORT", "8000"))

    # HTTP MCP endpoint будет: https://...onrender.com/mcp
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=port,
        path="/mcp",
    )

