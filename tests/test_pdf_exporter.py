"""Tests for the ReportLab Markdown-to-PDF exporter."""

import tempfile
import unittest
from pathlib import Path

import pymupdf

from agentx.reports import PDFExportError, PDFReportExporter


class PDFReportExporterTests(unittest.TestCase):
    def test_creates_readable_pdf_in_configured_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "pdfs"
            path = PDFReportExporter(output_dir).export(
                "# Research Report\n\nA concise report paragraph.",
                filename="research-report.pdf",
            )

            self.assertTrue(path.is_file())
            self.assertEqual(path.parent, output_dir)
            self.assertEqual(path.read_bytes()[:5], b"%PDF-")
            with pymupdf.open(path) as document:
                self.assertIn("Research Report", document[0].get_text())
                self.assertIn("A concise report paragraph.", document[0].get_text())

    def test_empty_markdown_is_rejected_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            exporter = PDFReportExporter(temp_dir)
            for report in ("", "  \n\t"):
                with self.subTest(report=repr(report)):
                    with self.assertRaisesRegex(ValueError, "must not be empty"):
                        exporter.export(report)

    def test_headings_lists_and_table_are_rendered(self):
        markdown = (
            "# Main Report\n\n## Findings\n\n"
            "- First finding\n- Second finding\n\n"
            "| Claim | Status |\n|---|---|\n| Finding A | supported |"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = PDFReportExporter(temp_dir).export(markdown)
            with pymupdf.open(path) as document:
                text = document[0].get_text()
            for expected in ("Main Report", "Findings", "First finding", "Second finding", "Claim", "supported"):
                with self.subTest(expected=expected):
                    self.assertIn(expected, text)

    def test_source_references_urls_and_document_pages_are_preserved(self):
        markdown = (
            "# Evidence Report\n\n## References and Citations\n\n"
            "- Web citation [1] — Study: https://example.test/source?a=1&b=2\n"
            "- Document citation [2] — report.pdf, page 17, chunk c17"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = PDFReportExporter(temp_dir).export(markdown)
            with pymupdf.open(path) as document:
                text = document[0].get_text()
            self.assertIn("Web citation [1]", text)
            self.assertIn("https://example.test/source?a=1&b=2", text)
            self.assertIn("report.pdf", text)
            self.assertIn("page 17", text)
            self.assertIn("chunk c17", text)

    def test_unsafe_filename_is_reduced_to_a_local_pdf_basename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "reports"
            path = PDFReportExporter(output_dir).export(
                "# Safe report", filename="..\\..\\outside folder\\report.pdf"
            )
            self.assertEqual(path.parent, output_dir)
            self.assertEqual(path.name, "report.pdf")

    def test_invalid_output_path_raises_export_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_parent = Path(temp_dir) / "file-not-directory"
            output_parent.write_text("occupied", encoding="utf-8")
            exporter = PDFReportExporter(output_parent / "child")
            with self.assertRaisesRegex(PDFExportError, "Could not write PDF report"):
                exporter.export("# A report")


if __name__ == "__main__":
    unittest.main()
