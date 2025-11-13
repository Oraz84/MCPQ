import os
import json
import asyncio
from typing import Dict, Any, List, Optional

from mcp.server import Server
from mcp.types import (
    InitializeRequest,
    InitializeResult,
    ToolsListResult,
    CallToolResult,
)

from openai import OpenAI
import httpx


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

server = Server("qdrant_rag")


async def embed(text: str) -> List[float]:
    resp = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return resp.data[0].embedding


async def qdrant_upsert(doc_id: str, vector: List[float], payload: Dict[str, Any]):
    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY
    }

    async with httpx.AsyncClient() as client:
        r = await client.put(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points",
            json={
                "points": [
                    {
                        "id": doc_id,
                        "vector": vector,
                        "payload": payload
                    }
                ]
            },
            headers=headers
        )
        r.raise_for_status()


async def qdrant_search(vec: List[float], top_k: int = 5):
    headers = {
        "Content-Type": "application/json",
        "api-key": QDRANT_API_KEY
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search",
            json={
                "vector": vec,
                "with_payload": True,
                "limit": top_k
            },
            headers=headers
        )
        r.raise_for_status()
        return r.json().get("result", [])


# ---------------- MCP ----------------

@server.initializer()
async def handle_initialize(req: InitializeRequest) -> InitializeResult:
    return InitializeResult(
        protocolVersion="2024-02-01",
        capabilities={"tools": True},
        sessionId="session-1"
    )


@server.list_tools()
async def handle_list_tools() -> ToolsListResult:
    return ToolsListResult(
        tools=[
            {
                "name": "store_document",
                "description": "Store document in Qdrant.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "text": {"type": "string"},
                        "metadata": {"type": "object"}
                    },
                    "required": ["doc_id", "text"]
                }
            },
            {
                "name": "search_documents",
                "description": "Search documents in Qdrant.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "number"}
                    },
                    "required": ["query"]
                }
            }
        ]
    )


@server.call_tool("store_document")
async def store_document(params: Dict[str, Any]) -> CallToolResult:
    doc_id = params["doc_id"]
    text = params["text"]
    metadata = params.get("metadata", {})
    vector = await embed(text)
    await qdrant_upsert(doc_id, vector, {"text": text, "metadata": metadata})
    return CallToolResult(output=f"Stored document {doc_id}")


@server.call_tool("search_documents")
async def search_docs(params: Dict[str, Any]) -> CallToolResult:
    query = params["query"]
    top_k = params.get("top_k", 5)
    vec = await embed(query)
    hits = await qdrant_search(vec, top_k)
    return CallToolResult(output=json.dumps(hits, indent=2))


# ---------------- RUN HTTP MCP ----------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    server.run_http("0.0.0.0", port, path="/mcp")
