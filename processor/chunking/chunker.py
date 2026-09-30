import re


def _split_markdown_sections(markdown: str) -> list[str]:
    """Partition markdown by headings while keeping each section self-contained."""
    if not markdown or not markdown.strip():
        return []

    lines = markdown.splitlines()
    sections: list[str] = []
    current: list[str] = []

    def flush() -> None:
        cleaned = "\n".join(current).strip()
        if cleaned:
            sections.append(cleaned)

    heading_re = re.compile(r"^(#{1,6})\s+.+$")

    for line in lines:
        if heading_re.match(line.strip()):
            if current:
                flush()
                current = []
            current.append(line)
            continue

        if line.startswith("|") and current and any(cell.startswith("|") for cell in current[-3:]):
            current.append(line)
            continue

        if not line.strip() and current:
            current.append("")
            continue

        if line.strip():
            current.append(line)

    flush()
    return sections


def chunk_markdown_sections(markdown: str, max_chars: int = 1800) -> list[str]:
    """Split markdown into heading-based sections and break oversized sections by paragraph."""
    sections = _split_markdown_sections(markdown)
    if not sections:
        return []

    chunks: list[str] = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section)
            continue

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if len(candidate) <= max_chars:
                current = candidate
                continue

            if current:
                chunks.append(current.strip())
                current = paragraph
            elif len(paragraph) <= max_chars:
                current = paragraph
            else:
                lines = paragraph.splitlines()
                wrapped: list[str] = []
                for line in lines:
                    if line.startswith("|") and (not wrapped or wrapped[-1].startswith("|")):
                        wrapped.append(line)
                        continue
                    if wrapped and "\n".join(wrapped).strip() and len("\n".join(wrapped) + "\n\n" + line) <= max_chars:
                        wrapped.append(line)
                    else:
                        if wrapped:
                            chunks.append("\n".join(wrapped).strip())
                        wrapped = [line]
                if wrapped:
                    chunks.append("\n".join(wrapped).strip())
        if current:
            chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """Keep the legacy chunking behavior for plain text while supporting markdown-aware sections."""
    if not text:
        return []

    if any(marker in text for marker in ("# ", "## ", "| ", "```")):
        return chunk_markdown_sections(text, max_chars=max(chunk_size, 1800))

    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(text), step):
        chunks.append(text[i : i + chunk_size])

    return chunks
