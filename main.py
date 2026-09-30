import sys
from pathlib import Path

from parser.pdf import PDFReader


def main() -> None:
    # Windows consoles default to cp1252, which can't encode characters like '≥'; force UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")

    sample_path = Path("C:\\Users\\VaibhavPrakash\\Downloads\\Satin & Tatami Stitch Embroidery SOP.pdf")
    with PDFReader(str(sample_path)) as reader:
        for page_num, text in enumerate(reader.iter_document()):
            print(f"\n--- EXTRACTING DATA FROM PAGE {page_num} ---")
            print(text)
        # The following FastAPI application code has been added
        from __future__ import annotations

        import json
        import re
        import tempfile
        from pathlib import Path
        from typing import Any

        from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
        from pydantic import BaseModel

        from ingestion.document_ingestor import DocumentIngestionService
        from parser.markdown import SUPPORTED_EXTENSIONS, convert_supported_file_to_markdown

        app = FastAPI(title="RAG Document Processor", version="1.0.0")
        ingestion_service = DocumentIngestionService()


        class HealthResponse(BaseModel):
            status: str


        class ConversionResponse(BaseModel):
            filename: str
            markdown: str


        class IngestionResponse(BaseModel):
            filename: str
            documents: list[dict[str, Any]]


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

if __name__ == "__main__":
    main()
