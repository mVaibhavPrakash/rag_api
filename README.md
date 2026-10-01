# Processor

Python service for parsing, chunking, and preparing document content for RAG indexing.

## Quick start

1. Copy `.env.example` to `.env` and fill in real `PINECONE_API_KEY` /
   `OPENAI_API_KEY` values. Keep `PINECONE_INDEX` lowercase
   alphanumeric/hyphen only (Pinecone rejects underscores or uppercase
   letters in index names).
2. Install dependencies with [uv](https://docs.astral.sh/uv/) (the project
   package itself doesn't need to be built, just its dependencies):

   ```bash
   uv sync --no-install-project
   ```

3. Start the API (module is `api.py` at the repo root, not under `src/`):

   ```bash
   # Windows
   .venv/Scripts/python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000

   # macOS/Linux
   .venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000
   ```

   Add `--reload` during development to auto-restart on code changes.

## API

- `GET /health` reports processor availability.
- `POST /convert` accepts multipart field `file` and returns normalized markdown.
- `POST /ingest` accepts multipart field `file` and optional `doc_id`, `namespace`,
  `doc_type`, comma-separated `applicable_tags`, and JSON-object `metadata_json`.

Point the web application at this service with `PYTHON_PROCESSOR_URL=http://localhost:8000`.
