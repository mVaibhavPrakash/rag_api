# Processor

Python service for parsing, chunking, and preparing document content for RAG indexing.

## Quick start

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000
```

## API

- `GET /health` reports processor availability.
- `POST /convert` accepts multipart field `file` and returns normalized markdown.
- `POST /ingest` accepts multipart field `file` and optional `doc_id`, `namespace`,
  `doc_type`, comma-separated `applicable_tags`, and JSON-object `metadata_json`.

Point the web application at this service with `PYTHON_PROCESSOR_URL=http://localhost:8000`.
