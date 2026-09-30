from __future__ import annotations

import re
from html import unescape
from pathlib import Path

from .pdf import PDFReader

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx"}


def _strip_html(raw_html: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", raw_html, flags=re.I | re.S)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|section|article|li|tr|h[1-6])>", "\n", text, flags=re.I)
    text = re.sub(r"<(?:p|div|section|article|li|tr|h[1-6])[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"</?t[dh][^>]*>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _markdown_from_html(raw_html: str) -> str:
    html = raw_html
    for level in range(1, 7):
        html = re.sub(
            rf"<h{level}\b[^>]*>(.*?)</h{level}>",
            lambda m: f"\n{'#' * level} {re.sub(r'<[^>]+>', ' ', m.group(1)).strip()}\n",
            html,
            flags=re.I | re.S,
        )

    text = _strip_html(html)
    if not text:
        return ""

    lines = [line.rstrip() for line in text.splitlines()]
    markdown_lines: list[str] = []
    in_list = False

    for line in lines:
        if not line.strip():
            if in_list:
                markdown_lines.append("")
            continue

        stripped = line.strip()
        if stripped.startswith("#"):
            markdown_lines.append(stripped)
            in_list = False
            continue

        if re.fullmatch(r"[\-\*]\s+.+", stripped):
            markdown_lines.append(f"- {stripped[2:].strip()}")
            in_list = True
            continue

        markdown_lines.append(stripped)
        in_list = False

    return "\n\n".join(part.strip() for part in "\n".join(markdown_lines).split("\n\n") if part.strip())


def _markdown_from_txt(raw_text: str, source_name: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    first_line = lines[0].strip()
    if first_line.startswith("#"):
        return text

    if first_line.lower().startswith("title:"):
        heading = first_line.split(":", 1)[1].strip() or source_name
        return f"# {heading}\n\n" + "\n\n".join(lines[1:])

    if len(lines) > 1:
        return f"# {first_line}\n\n" + "\n\n".join(lines[1:])

    return f"# {source_name}\n\n{text}"


def _markdown_from_pdf(path: Path) -> str:
    with PDFReader(str(path)) as reader:
        pages = reader.extract_document()
    return "\n\n".join(page.strip() for page in pages if page and page.strip())


def _markdown_from_docx(path: Path) -> str:
    try:
        import docx  # type: ignore
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError("docx conversion requires the python-docx package to be installed.") from exc

    document = docx.Document(str(path))
    blocks: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        blocks.append(text)
    return "\n\n".join(blocks)


def convert_supported_file_to_markdown(file_path: str | Path) -> str:
    """Read a supported source document and normalize it to markdown.

    Manual section markdown files remain the recommended authoring model for SOPs.
    This function is meant for converting uploaded/raw source files (txt, html, pdf,
    docx) into markdown before downstream chunking.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return path.read_text(encoding="utf-8")

    if suffix in {".txt"}:
        return _markdown_from_txt(path.read_text(encoding="utf-8"), path.stem.replace("_", " ").strip() or path.name)

    if suffix in {".html", ".htm"}:
        return _markdown_from_html(path.read_text(encoding="utf-8"))

    if suffix == ".pdf":
        return _markdown_from_pdf(path)

    if suffix == ".docx":
        return _markdown_from_docx(path)

    raise ValueError(f"Unsupported file type: {suffix or path.name}. Supported types: {sorted(SUPPORTED_EXTENSIONS)}")


__all__ = ["SUPPORTED_EXTENSIONS", "convert_supported_file_to_markdown"]
