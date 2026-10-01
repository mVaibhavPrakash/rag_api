from __future__ import annotations

import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from processor.ingestion.document_ingestor import DocumentIngestionService
from processor.parser.markdown import SUPPORTED_EXTENSIONS, convert_supported_file_to_markdown

load_dotenv()

app = FastAPI(title="RAG Document Processor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8765").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ingestion_service = DocumentIngestionService()


# ---------------------------------------------------------------------------
# Lazy service factories (only constructed when the first request arrives so
# missing env vars don't crash startup for services that aren't in use)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _vector_service():
    from processor.vector.vector import VectorService  # noqa: PLC0415

    return VectorService()


@lru_cache(maxsize=1)
def _retriever_service():
    from processor.retriever.retriever import RetrieverService  # noqa: PLC0415

    return RetrieverService()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str


class ConversionResponse(BaseModel):
    filename: str
    markdown: str


class IngestionResponse(BaseModel):
    filename: str
    documents: list[dict[str, Any]]


# -- Vector --


class UpsertRequest(BaseModel):
    records: list[dict[str, Any]]
    namespace: str | None = None


class UpsertResponse(BaseModel):
    upserted_count: int


class DeleteRequest(BaseModel):
    doc_ids: list[str]
    namespace: str | None = None


# -- Retriever --


class RetrieveRequest(BaseModel):
    query: str
    namespace: str | None = None
    top_k: int | None = None
    filter: dict[str, Any] | None = None
    include_chunks: bool = True


class ChunkResponse(BaseModel):
    doc_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class RetrieveResponse(BaseModel):
    query: str
    answer: str
    model: str
    usage: dict[str, int]
    chunks: list[ChunkResponse]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _safe_filename(filename: str | None) -> str:
    safe_name = Path(filename or "upload.txt").name
    if not safe_name or safe_name == ".":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="A valid filename is required.")
    if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Supported types: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    return safe_name


def _default_doc_id(filename: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9]+", "-", Path(filename).stem).strip("-").upper()
    return identifier or "DOCUMENT"


def _parse_metadata(metadata_json: str | None) -> dict[str, Any]:
    if not metadata_json:
        return {}
    try:
        value = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="metadata_json must contain a JSON object.") from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="metadata_json must contain a JSON object.")
    return value


async def _save_upload(upload: UploadFile) -> tuple[tempfile.TemporaryDirectory[str], Path, str]:
    filename = _safe_filename(upload.filename)
    temporary_directory = tempfile.TemporaryDirectory()
    path = Path(temporary_directory.name) / filename
    path.write_bytes(await upload.read())
    return temporary_directory, path, filename


# ---------------------------------------------------------------------------
# Routes — general
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/convert", response_model=ConversionResponse)
async def convert_document(file: UploadFile = File(...)) -> ConversionResponse:
    temporary_directory, path, filename = await _save_upload(file)
    try:
        return ConversionResponse(filename=filename, markdown=convert_supported_file_to_markdown(path))
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    finally:
        temporary_directory.cleanup()


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    file: UploadFile = File(...),
    doc_id: str | None = Form(default=None),
    namespace: str | None = Form(default=None),
    doc_type: str | None = Form(default=None),
    applicable_tags: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
) -> IngestionResponse:
    temporary_directory, path, filename = await _save_upload(file)
    try:
        tags = [tag.strip() for tag in (applicable_tags or "").split(",") if tag.strip()]
        records = ingestion_service.ingest_file(
            path,
            doc_id=doc_id or _default_doc_id(filename),
            namespace=namespace,
            doc_type=doc_type,
            applicable_tags=tags,
            metadata=_parse_metadata(metadata_json),
        )
        return IngestionResponse(filename=filename, documents=records)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    finally:
        temporary_directory.cleanup()


# ---------------------------------------------------------------------------
# Routes — vector service
# ---------------------------------------------------------------------------


@app.post("/vector/upsert", response_model=UpsertResponse, tags=["vector"])
def vector_upsert(body: UpsertRequest) -> UpsertResponse:
    """Embed and upsert a list of index records (output of ``/ingest``) into Pinecone."""
    try:
        result = _vector_service().upsert_records(body.records, namespace=body.namespace)
        return UpsertResponse(upserted_count=result["upserted_count"])
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing required environment variable: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@app.delete("/vector/delete", tags=["vector"])
def vector_delete(body: DeleteRequest) -> dict[str, str]:
    """Delete vectors from Pinecone by their IDs."""
    try:
        _vector_service().delete_records(body.doc_ids, namespace=body.namespace)
        return {"status": "deleted"}
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Routes — retriever service
# ---------------------------------------------------------------------------


@app.post("/retrieve", response_model=RetrieveResponse, tags=["retriever"])
def retrieve(body: RetrieveRequest) -> RetrieveResponse:
    """Query the vector database and generate a grounded answer via OpenAI."""
    try:
        result = _retriever_service().retrieve_and_answer(
            body.query,
            namespace=body.namespace,
            top_k=body.top_k,
            filter=body.filter,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing required environment variable: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    chunks = (
        [
            ChunkResponse(
                doc_id=c.doc_id,
                score=c.score,
                text=c.text,
                metadata=c.metadata,
            )
            for c in result.chunks
        ]
        if body.include_chunks
        else []
    )

    return RetrieveResponse(
        query=result.query,
        answer=result.answer,
        model=result.model,
        usage=result.usage,
        chunks=chunks,
    )


@app.post("/retrieve/chunks", response_model=list[ChunkResponse], tags=["retriever"])
def retrieve_chunks_only(body: RetrieveRequest) -> list[ChunkResponse]:
    """Return raw vector-search results without calling the LLM."""
    try:
        chunks = _retriever_service().retrieve_only(
            body.query,
            namespace=body.namespace,
            top_k=body.top_k,
            filter=body.filter,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing required environment variable: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return [
        ChunkResponse(doc_id=c.doc_id, score=c.score, text=c.text, metadata=c.metadata)
        for c in chunks
    ]


# ---------------------------------------------------------------------------
# Routes — frontend-facing composite endpoints
# ---------------------------------------------------------------------------
# These thin wrappers match the paths the React SPA calls so the Vite proxy
# can forward /api/* straight to this server without any path rewriting.


class DocumentsUploadResponse(BaseModel):
    documents: list[dict[str, Any]]


class ChatRequest(BaseModel):
    question: str
    namespace: str | None = None
    filter: dict[str, Any] | None = None
    top_k: int | None = None


class ChatResponse(BaseModel):
    answer: str


@app.post("/api/documents", response_model=DocumentsUploadResponse, tags=["frontend"])
async def api_documents(
    file: UploadFile = File(...),
    doc_id: str | None = Form(default=None),
    namespace: str | None = Form(default=None),
    doc_type: str | None = Form(default=None),
    applicable_tags: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
) -> DocumentsUploadResponse:
    """Ingest a document and immediately upsert its chunks into Pinecone.

    This is the single endpoint the frontend calls when the user saves a file.
    It chains /ingest → /vector/upsert internally so the browser never touches
    the vector DB directly.
    """
    temporary_directory, path, filename = await _save_upload(file)
    try:
        tags = [t.strip() for t in (applicable_tags or "").split(",") if t.strip()]
        effective_doc_id = doc_id or _default_doc_id(filename)
        records = ingestion_service.ingest_file(
            path,
            doc_id=effective_doc_id,
            namespace=namespace,
            doc_type=doc_type,
            applicable_tags=tags,
            metadata=_parse_metadata(metadata_json),
        )

        # Upsert straight into Pinecone
        _vector_service().upsert_records(records, namespace=namespace)

        metadata_flat = _parse_metadata(metadata_json)
        doc_entry = {
            "name": filename,
            "doc_id": effective_doc_id,
            "namespace": namespace or "",
            "doc_type": doc_type or "",
            "metadata": metadata_flat,
            "chunk_count": len(records),
        }
        return DocumentsUploadResponse(documents=[doc_entry])
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    finally:
        temporary_directory.cleanup()


@app.post("/api/chat", response_model=ChatResponse, tags=["frontend"])
def api_chat(body: ChatRequest) -> ChatResponse:
    """Run hybrid RAG retrieval + generation and return just the answer string.

    This is the single endpoint the frontend calls from the chat panel.
    """
    try:
        result = _retriever_service().retrieve_and_answer(
            body.question,
            namespace=body.namespace,
            top_k=body.top_k,
            filter=body.filter,
        )
        return ChatResponse(answer=result.answer)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing required environment variable: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
