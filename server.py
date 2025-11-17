import os
import json
from typing import Dict, Any, List

from mcp.server import Server
from mcp.types import InitializeResult

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)

import google.oauth2.service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

import docx
import PyPDF2

GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID", "")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION = "rag_docs"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

server = Server("fh_mcp")


def get_gdrive_service():
    creds_info = json.loads(GOOGLE_CREDS_JSON)
    creds = google.oauth2.service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)


def extract_text_from_file(file_bytes: bytes, mime_type: str):
    if mime_type == "application/pdf":
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    if mime_type.startswith("text/"):
        return file_bytes.decode("utf-8", errors="ignore")
    return ""


qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

if not qdrant.collection_exists(QDRANT_COLLECTION):
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE
        )
    )


async def get_embedding(text: str) -> List[float]:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": "text-embedding-3-small", "input": text}
        )
        j = r.json()
        return j["data"][0]["embedding"]


@server.on("initialize")
async def on_initialize(request, response):
    return InitializeResult(
        protocolVersion="2024-02-01",
        capabilities={"tools": True},
        sessionId="fh-session"
    )


@server.on("tools/list")
async def on_tools_list(request, response):
    return {
        "tools": [
            {
                "name": "gdrive_search",
                "description": "Search files in Google Drive folder",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            },
            {
                "name": "gdrive_read",
                "description": "Read & extract text from Google Drive file",
                "inputSchema": {
                    "type": "object",
                    "properties": {"file_id": {"type": "string"}},
                    "required": ["file_id"]
                }
            },
            {
                "name": "rag_upsert",
                "description": "Add document text to Qdrant RAG",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "text": {"type": "string"}
                    },
                    "required": ["doc_id", "text"]
                }
            },
            {
                "name": "rag_search",
                "description": "Search Qdrant RAG",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]
    }


@server.on("tools/call")
async def on_tools_call(request, response):
    tool = request.params["name"]
    args = request.params.get("arguments", {})

    if tool == "gdrive_search":
        query = args["query"]
        service = get_gdrive_service()
        res = service.files().list(
            q=f"'{GDRIVE_FOLDER_ID}' in parents and trashed=false and name contains '{query}'",
            fields="files(id, name, mimeType)"
        ).execute()
        return {"files": res.get("files", [])}

    if tool == "gdrive_read":
        file_id = args["file_id"]
        service = get_gdrive_service()
        meta = service.files().get(fileId=file_id, fields="id,name,mimeType").execute()
        mime = meta["mimeType"]

        if mime.startswith("application/vnd.google-apps"):
            req = service.files().export_media(fileId=file_id, mimeType="application/pdf")
        else:
            req = service.files().get_media(fileId=file_id)

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        text = extract_text_from_file(fh.getvalue(), mime)
        return {"text": text}

    if tool == "rag_upsert":
        doc_id = args["doc_id"]
        text = args["text"]
        emb = await get_embedding(text)
        qdrant.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=doc_id,
                    vector=emb,
                    payload={"text": text}
                )
            ]
        )
        return {"status": "ok"}

    if tool == "rag_search":
        query = args["query"]
        emb = await get_embedding(query)
        hits = qdrant.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=emb,
            limit=5
        )
        return {
            "results": [
                {
                    "score": h.score,
                    "text": h.payload.get("text", "")
                }
                for h in hits
            ]
        }

    # NEW: correct MCP-style error response
    return {
        "error": {
            "code": -1,
            "message": f"Unknown tool: {tool}"
        }
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    server.run_http("0.0.0.0", port, path="/mcp")
