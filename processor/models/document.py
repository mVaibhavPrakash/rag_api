from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    title: str
    content: str
    namespace: str | None = None
    doc_type: str | None = None
    is_shared: bool | None = None
    applicable_tags: list[str] = field(default_factory=list)
    section_title: str | None = None
    parent_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_index_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "doc_id": self.id,
            "section_title": self.section_title or self.title,
            "parent_text": self.parent_text or self.content,
        }

        if self.namespace is not None:
            record["namespace"] = self.namespace
        if self.doc_type is not None:
            record["doc_type"] = self.doc_type
        if self.is_shared is not None:
            record["is_shared"] = self.is_shared
        if self.applicable_tags:
            record["applicable_tags"] = self.applicable_tags

        record.update(self.metadata)
        return record
