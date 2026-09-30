import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from chunking.chunker import chunk_markdown_sections
from ingestion.document_ingestor import DocumentIngestionService
from api import app
from models.document import Document
from parser.markdown import convert_supported_file_to_markdown


class ChunkerTests(unittest.TestCase):
    def test_processor_api_converts_and_ingests_uploaded_text(self):
        client = TestClient(app)
        upload = {"file": ("leave_policy.txt", b"# Maternity Leave\n\nEmployees may take 12 weeks.", "text/plain")}

        conversion = client.post("/convert", files=upload)
        self.assertEqual(conversion.status_code, 200)
        self.assertIn("# Maternity Leave", conversion.json()["markdown"])

        ingestion = client.post(
            "/ingest",
            files=upload,
            data={
                "namespace": "human_resources",
                "doc_type": "Policy",
                "applicable_tags": "Leave,Full-Time",
                "metadata_json": '{"owner":"HR"}',
            },
        )
        self.assertEqual(ingestion.status_code, 200)
        record = ingestion.json()["documents"][0]
        self.assertEqual(record["namespace"], "human_resources")
        self.assertEqual(record["owner"], "HR")

    def test_keeps_tables_intact_and_groups_headings(self):
        markdown = """
# Hat SOP

## Universal rules

Needle size is 75/11.

| Band | Needle |
| --- | --- |
| 1 | 75/11 |
| 2 | 80/12 |

## Polo rules

Use breathable settings.
""".strip()

        chunks = chunk_markdown_sections(markdown)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(any("Universal rules" in chunk for chunk in chunks))
        self.assertTrue(any("| Band | Needle |" in chunk for chunk in chunks))
        self.assertTrue(all("| Band | Needle |" not in chunk or chunk.count("| Band | Needle |") == 1 for chunk in chunks))
        self.assertTrue(any("Polo rules" in chunk for chunk in chunks))

    def test_converts_plain_text_and_html_to_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "sample.txt"
            txt_path.write_text("Title\n\nNeedle size is 75/11.", encoding="utf-8")

            html_path = Path(tmpdir) / "sample.html"
            html_path.write_text("<html><body><h1>Hat SOP</h1><p>Use breathable settings.</p></body></html>", encoding="utf-8")

            txt_markdown = convert_supported_file_to_markdown(txt_path)
            html_markdown = convert_supported_file_to_markdown(html_path)

            self.assertIn("# Title", txt_markdown)
            self.assertIn("Needle size is 75/11.", txt_markdown)
            self.assertIn("# Hat SOP", html_markdown)
            self.assertIn("Use breathable settings.", html_markdown)

    def test_document_record_supports_domain_specific_metadata(self):
        document = Document(
            id="hr-leave-001",
            title="Maternity Leave",
            content="Policy details",
            namespace="human_resources",
            doc_type="Policy",
            is_shared=False,
            applicable_tags=["Leave", "Full-Time"],
            metadata={"owner": "HR", "region": "global"},
        )

        record = document.to_index_record()

        self.assertEqual(record["namespace"], "human_resources")
        self.assertEqual(record["doc_type"], "Policy")
        self.assertEqual(record["applicable_tags"], ["Leave", "Full-Time"])
        self.assertEqual(record["owner"], "HR")
        self.assertEqual(record["region"], "global")

    def test_document_ingestor_converts_files_and_emits_index_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "leave_policy.txt"
            source_path.write_text("# Maternity Leave\n\nEmployees may take 12 weeks of paid leave.", encoding="utf-8")

            records = DocumentIngestionService().ingest_file(
                source_path,
                doc_id="hr-leave-policy",
                namespace="human_resources",
                doc_type="Policy",
                applicable_tags=["Leave", "Full-Time"],
                metadata={"owner": "HR"},
            )

            self.assertTrue(records)
            self.assertEqual(records[0]["namespace"], "human_resources")
            self.assertEqual(records[0]["doc_type"], "Policy")
            self.assertIn("Maternity Leave", records[0]["section_title"])
            self.assertIn("Employees may take", records[0]["parent_text"])
            self.assertEqual(records[0]["owner"], "HR")


if __name__ == "__main__":
    unittest.main()
