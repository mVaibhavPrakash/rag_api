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
   ```

   Add `--reload` during development to auto-restart on code changes.

## API

- `GET /health` reports processor availability.
- `POST /convert` accepts multipart field `file` and returns normalized markdown.
- `POST /ingest` accepts multipart field `file` and optional `doc_id`, `namespace`,
  `doc_type`, comma-separated `applicable_tags`, and JSON-object `metadata_json`.

Point the web application at this service with `PYTHON_PROCESSOR_URL=http://localhost:8000`.

## Troubleshooting

### Port 8000 is already in use

If you see an error like:

```text
ERROR: [Errno 10048] error while attempting to bind on address
('0.0.0.0', 8000): [winerror 10048]
Only one usage of each socket address (protocol/network address/port) is normally permitted
```

This means another process is already using port `8000`.

#### 1. Find the process using port 8000

In PowerShell, run:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen
```

Example:

```text
LocalAddress LocalPort RemoteAddress RemotePort State  OwningProcess
------------ --------- ------------- ---------- -----  -------------
0.0.0.0      8000      0.0.0.0       0          Listen 26976
```

Note the `OwningProcess` PID.

#### 2. Check the process

Replace `<PID>` with the PID from the previous command:

```powershell
Get-Process -Id <PID>
```

For example:

```powershell
Get-Process -Id 26976
```

If it is an old Python/Uvicorn process, stop it:

```powershell
Stop-Process -Id <PID> -Force
```

Then start the API again:

```powershell
.venv/Scripts/python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000
```

#### Alternative: Use a different port

If you don't want to stop the existing process, start the API on another port:

```powershell
.venv/Scripts/python.exe -m uvicorn api:app --host 0.0.0.0 --port 8001
```

The API will then be available at:

```text
http://localhost:8001
```

> **Note:** If you are using Git Bash, commands such as `tasklist /FI` and `taskkill /PID` may not work as expected because Git Bash can interpret Windows command-line options as paths. PowerShell is recommended for these troubleshooting commands.
