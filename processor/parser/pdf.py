from collections.abc import Iterator

import pdfplumber


def _serialize_table(rows: list[list[str | None]]) -> str:
    # Render table rows as a markdown table so it stays inline with the surrounding prose.
    cleaned = [
        [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
        for row in rows
    ]
    if not cleaned:
        return ""

    header, *body = cleaned
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def extract_page_blocks(page) -> list[dict]:
    # Merge a page's text lines and tables into one top-to-bottom reading-order sequence.
    table_objects = page.find_tables()

    # Crop tables out before pulling text lines so cell text doesn't leak into the surrounding prose
    text_page = page
    for table_obj in table_objects:
        text_page = text_page.outside_bbox(table_obj.bbox)

    blocks = [
        {"type": "text", "top": line["top"], "content": line["text"]}
        for line in text_page.extract_text_lines()
        if line["text"].strip()
    ]
    blocks += [
        {"type": "table", "top": table_obj.bbox[1], "content": _serialize_table(table_obj.extract())}
        for table_obj in table_objects
    ]

    # Sort by vertical position so tables land back where they occurred relative to the text
    blocks.sort(key=lambda block: block["top"])
    return blocks


class PDFReader:
    # Thin, resource-safe wrapper around an open pdfplumber PDF.
    # Pass `pages` (1-indexed page numbers) to only load a subset of a large document.
    def __init__(self, pdf_path: str, pages: list[int] | None = None):
        self.pdf_path = pdf_path
        self.pdf = pdfplumber.open(pdf_path, pages=pages)

    @property
    def page_count(self) -> int:
        return len(self.pdf.pages)

    def close(self) -> None:
        self.pdf.close()

    def __enter__(self) -> "PDFReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort safety net only; __del__ isn't guaranteed to run, so this
        # doesn't replace using `with PDFReader(...)` or calling close() explicitly.
        try:
            self.close()
        except Exception:
            pass

    def extract_page_data(self, page_num: int = 0) -> str:
        # Extract one page's text and tables interleaved in their original reading order.
        if page_num >= self.page_count:
            raise IndexError(f"page_num {page_num} out of range (0-{self.page_count - 1})")

        page = self.pdf.pages[page_num]
        blocks = extract_page_blocks(page)
        content = "\n\n".join(block["content"] for block in blocks)

        # Drop this page's parsed chars/lines/rects now that we're done with it,
        # otherwise every page's geometry stays cached in memory for the PDF's lifetime.
        page.flush_cache()
        return content

    def iter_document(self) -> Iterator[str]:
        # Stream every page's ordered content one page at a time, bounding memory to ~1 page.
        for page_num in range(self.page_count):
            yield self.extract_page_data(page_num)

    def extract_document(self) -> list[str]:
        # Eagerly collect every page's content. Prefer `iter_document` for large PDFs.
        return list(self.iter_document())
