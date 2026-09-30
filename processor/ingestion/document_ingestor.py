from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from apps.processor.chunking.chunker import chunk_markdown_sections
from apps.processor.models.document import Document
from apps.processor.parser.markdown import convert_supported_file_to_markdown


class DocumentIngestionService:
    """Convert uploaded source files into markdown sections and Pinecone-ready records."""

    def __init__(self, chunk_size: int = 1800):
        self.chunk_size = chunk_size

    @staticmethod
    def _extract_heading(markdown: str) -> str | None:
        match = re.search(r"^#{1,6}\s+(.+)$", markdown, flags=re.MULTILINE)
        if match is None:
            return None
        return match.group(1).strip()

    def ingest_file(
        self,
        file_path: str | Path,
        *,
        doc_id: str,
        namespace: str | None = None,
        doc_type: str | None = None,
        applicable_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        section_title: str | None = None,
    ) -> list[dict[str, Any]]:
        path = Path(file_path)
        markdown = convert_supported_file_to_markdown(path)
        chunks = chunk_markdown_sections(markdown, max_chars=self.chunk_size)

        if not chunks:
            chunks = [markdown.strip()] if markdown.strip() else []

        records: list[dict[str, Any]] = []
        for chunk_index, chunk in enumerate(chunks):
            chunk_title = self._extract_heading(chunk) or section_title or path.stem.replace("_", " ").strip() or path.name
            document = Document(
                id=f"{doc_id}:{chunk_index}",
                title=chunk_title,
                content=chunk,
                namespace=namespace,
                doc_type=doc_type,
                applicable_tags=list(applicable_tags or []),
                section_title=chunk_title,
                parent_text=chunk,
                metadata=dict(metadata or {}),
            )
            records.append(document.to_index_record())

        return records
