"""Export AgentX Markdown reports to readable, locally stored PDFs."""

from __future__ import annotations

import re
from datetime import datetime
from html import escape, unescape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class PDFExportError(RuntimeError):
    """Raised when a Markdown report cannot be written as a PDF."""


class PDFReportExporter:
    """Render AgentX report Markdown without changing its factual content.

    The default output directory is ``reports/generated`` in the current
    project. It is ignored by Git in the AgentX project configuration.
    """

    DEFAULT_OUTPUT_DIR = Path("reports") / "generated"
    _registered_fonts = False
    _heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    _link_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
    _url_pattern = re.compile(r"https?://[^\s<>]+")
    _special_start = re.compile(r"^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s?|\|.*\||```|---+$)")

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else self.DEFAULT_OUTPUT_DIR
        self._register_fonts()
        self.styles = self._make_styles()

    def export(self, markdown: str, *, filename: str | None = None) -> Path:
        """Write Markdown to PDF and return its path.

        Empty or whitespace-only Markdown is rejected with ``ValueError``.
        Filesystem and ReportLab errors are wrapped in ``PDFExportError``.
        """
        if not isinstance(markdown, str) or not markdown.strip():
            raise ValueError("Markdown report must not be empty.")

        output_name = self._safe_filename(filename or self._filename_from_report(markdown))
        output_path = self.output_dir / output_name
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            document = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=22 * mm,
                leftMargin=22 * mm,
                topMargin=24 * mm,
                bottomMargin=22 * mm,
                title="AgentX Research Report",
                author="AgentX",
                subject="Evidence-aware research report",
            )
            document.build(
                self._build_story(markdown),
                onFirstPage=self._draw_page_chrome,
                onLaterPages=self._draw_page_chrome,
            )
        except (OSError, ValueError, TypeError) as exc:
            raise PDFExportError(f"Could not write PDF report to '{output_path}': {exc}") from exc
        except Exception as exc:
            raise PDFExportError(
                f"PDF generation failed for '{output_path}' ({type(exc).__name__}): {exc}"
            ) from exc
        return output_path

    @classmethod
    def _safe_filename(cls, requested: str) -> str:
        """Reduce a requested name to a portable basename inside output_dir."""
        leaf = requested.replace("\\", "/").rsplit("/", 1)[-1]
        stem = Path(leaf).stem
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(" .-_ ")
        stem = re.sub(r"-{2,}", "-", stem)[:100].rstrip(" .-")
        if not stem:
            stem = "agentx-report"
        if stem.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
            stem = f"report-{stem}"
        return f"{stem}.pdf"

    @classmethod
    def _filename_from_report(cls, markdown: str) -> str:
        for line in markdown.splitlines():
            match = cls._heading_pattern.match(line.strip())
            if match:
                return f"{match.group(2)}-{datetime.now():%Y%m%d-%H%M%S}.pdf"
        return f"agentx-report-{datetime.now():%Y%m%d-%H%M%S}.pdf"

    @classmethod
    def _register_fonts(cls) -> None:
        if cls._registered_fonts:
            return
        try:
            import reportlab

            font_dir = Path(reportlab.__file__).parent / "fonts"
            font_files = {
                "AgentXSans": "Vera.ttf",
                "AgentXSans-Bold": "VeraBd.ttf",
                "AgentXSans-Italic": "VeraIt.ttf",
                "AgentXSans-BoldItalic": "VeraBI.ttf",
            }
            for name, filename in font_files.items():
                path = font_dir / filename
                if path.is_file():
                    pdfmetrics.registerFont(TTFont(name, str(path)))
            if "AgentXSans" in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFontFamily(
                    "AgentXSans",
                    normal="AgentXSans",
                    bold="AgentXSans-Bold",
                    italic="AgentXSans-Italic",
                    boldItalic="AgentXSans-BoldItalic",
                )
        except (ImportError, OSError):
            # Standard PDF fonts remain a usable fallback if bundled Vera
            # fonts are unavailable in a constrained ReportLab distribution.
            pass
        cls._registered_fonts = True

    @staticmethod
    def _font_name() -> str:
        return "AgentXSans" if "AgentXSans" in pdfmetrics.getRegisteredFontNames() else "Helvetica"

    def _make_styles(self) -> dict[str, ParagraphStyle]:
        base = getSampleStyleSheet()
        font = self._font_name()
        bold = "AgentXSans-Bold" if font == "AgentXSans" else "Helvetica-Bold"
        return {
            "body": ParagraphStyle(
                "AgentXBody", parent=base["BodyText"], fontName=font,
                fontSize=9.5, leading=14, textColor=colors.HexColor("#263238"),
                spaceAfter=6, splitLongWords=1,
            ),
            "title": ParagraphStyle(
                "AgentXTitle", parent=base["Title"], fontName=bold,
                fontSize=22, leading=27, textColor=colors.HexColor("#16324F"),
                alignment=TA_LEFT, spaceAfter=15,
            ),
            "h2": ParagraphStyle(
                "AgentXH2", parent=base["Heading2"], fontName=bold,
                fontSize=15, leading=19, textColor=colors.HexColor("#176B87"),
                spaceBefore=12, spaceAfter=6, keepWithNext=True,
            ),
            "h3": ParagraphStyle(
                "AgentXH3", parent=base["Heading3"], fontName=bold,
                fontSize=11.5, leading=15, textColor=colors.HexColor("#274C5E"),
                spaceBefore=9, spaceAfter=4, keepWithNext=True,
            ),
            "small": ParagraphStyle(
                "AgentXSmall", parent=base["BodyText"], fontName=font,
                fontSize=8.5, leading=11, splitLongWords=1,
            ),
            "table_header": ParagraphStyle(
                "AgentXTableHeader", parent=base["BodyText"], fontName=bold,
                fontSize=8.5, leading=11, textColor=colors.white,
            ),
            "quote": ParagraphStyle(
                "AgentXQuote", parent=base["BodyText"], fontName=font,
                fontSize=9, leading=13, leftIndent=10, borderColor=colors.HexColor("#9BB7C5"),
                borderWidth=0, textColor=colors.HexColor("#3F515A"), spaceAfter=5,
            ),
        }

    def _build_story(self, markdown: str) -> list[object]:
        lines = markdown.splitlines()
        story: list[object] = []
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if not line:
                index += 1
                continue
            heading = self._heading_pattern.match(line)
            if heading:
                level = len(heading.group(1))
                key = "title" if level == 1 else "h2" if level == 2 else "h3"
                story.append(Paragraph(self._inline(heading.group(2)), self.styles[key]))
                index += 1
                continue
            if line.startswith("```"):
                code: list[str] = []
                index += 1
                while index < len(lines) and not lines[index].strip().startswith("```"):
                    code.append(lines[index])
                    index += 1
                if index < len(lines):
                    index += 1
                story.append(Preformatted("\n".join(code), self.styles["small"]))
                story.append(Spacer(1, 5))
                continue
            if self._is_table_start(lines, index):
                table, index = self._parse_table(lines, index)
                story.append(table)
                story.append(Spacer(1, 7))
                continue
            if line.startswith(("- ", "* ", "+ ")):
                items, index = self._parse_list(lines, index, ordered=False)
                story.append(ListFlowable(
                    [ListItem(Paragraph(item, self.styles["body"]), leftIndent=10) for item in items],
                    bulletType="bullet", start="circle", leftIndent=17, bulletFontName=self._font_name(),
                    bulletFontSize=7, spaceAfter=5,
                ))
                continue
            if re.match(r"^\d+[.)]\s+", line):
                items, index = self._parse_list(lines, index, ordered=True)
                story.append(ListFlowable(
                    [ListItem(Paragraph(item, self.styles["body"]), leftIndent=10) for item in items],
                    bulletType="1", leftIndent=20, bulletFontName=self._font_name(),
                    bulletFontSize=8, spaceAfter=5,
                ))
                continue
            if line.startswith(">"):
                quote_lines: list[str] = []
                while index < len(lines) and lines[index].strip().startswith(">"):
                    quote_lines.append(re.sub(r"^>\s?", "", lines[index].strip()))
                    index += 1
                story.append(Paragraph(self._inline(" ".join(quote_lines)), self.styles["quote"]))
                continue
            if line.startswith("---"):
                story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#B8C7CE"), spaceBefore=5, spaceAfter=8))
                index += 1
                continue

            paragraph_lines = [line]
            index += 1
            while index < len(lines):
                candidate = lines[index].strip()
                if not candidate or self._special_start.match(candidate):
                    break
                paragraph_lines.append(candidate)
                index += 1
            story.append(Paragraph(self._inline(" ".join(paragraph_lines)), self.styles["body"]))
        return story

    @staticmethod
    def _is_table_start(lines: list[str], index: int) -> bool:
        return (
            index + 1 < len(lines)
            and "|" in lines[index]
            and re.fullmatch(r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*", lines[index + 1]) is not None
        )

    def _parse_table(self, lines: list[str], index: int) -> tuple[Table, int]:
        rows: list[list[str]] = []
        while index < len(lines) and "|" in lines[index].strip():
            row = self._table_cells(lines[index])
            if not all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in row):
                rows.append(row)
            index += 1
        column_count = max((len(row) for row in rows), default=1)
        rows = [row + [""] * (column_count - len(row)) for row in rows]
        rendered = [
            [Paragraph(self._inline(cell), self.styles["table_header"] if row_index == 0 else self.styles["small"]) for cell in row]
            for row_index, row in enumerate(rows)
        ]
        table = Table(rendered, repeatRows=1, hAlign=TA_LEFT, colWidths=None)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176B87")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C7CE")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F8")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return table, index

    @staticmethod
    def _table_cells(line: str) -> list[str]:
        cells = line.strip().strip("|").split("|")
        return [cell.strip() for cell in cells]

    def _parse_list(self, lines: list[str], index: int, *, ordered: bool) -> tuple[list[str], int]:
        items: list[str] = []
        pattern = re.compile(r"^\d+[.)]\s+(.*)$") if ordered else re.compile(r"^[-*+]\s+(.*)$")
        while index < len(lines):
            match = pattern.match(lines[index].strip())
            if not match:
                break
            items.append(self._inline(match.group(1)))
            index += 1
        return items, index

    def _inline(self, text: str) -> str:
        """Translate the small inline Markdown subset used by AgentX reports."""
        links: list[str] = []

        def reserve_link(match: re.Match[str]) -> str:
            label, url = match.groups()
            escaped_url = escape(url, {'"': "&quot;"})
            links.append(f'<link href="{escaped_url}" color="#176B87">{escape(label)}</link>')
            return f"AGENTXLINKTOKEN{len(links) - 1}END"

        text = self._link_pattern.sub(reserve_link, text)
        text = escape(text)
        text = self._url_pattern.sub(self._raw_url_link, text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
        text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
        for index, link in enumerate(links):
            text = text.replace(f"AGENTXLINKTOKEN{index}END", link)
        return text

    @staticmethod
    def _raw_url_link(match: re.Match[str]) -> str:
        value = unescape(match.group(0))
        suffix = ""
        while value and value[-1] in ".,;:!?)":
            suffix = value[-1] + suffix
            value = value[:-1]
        href = escape(value, {'"': "&quot;"})
        return f'<link href="{href}" color="#176B87">{escape(value)}</link>{suffix}'

    def _draw_page_chrome(self, canvas: object, document: object) -> None:
        canvas.saveState()
        width, _ = A4
        canvas.setStrokeColor(colors.HexColor("#D5E0E5"))
        canvas.setLineWidth(0.5)
        canvas.line(22 * mm, 16 * mm, width - 22 * mm, 16 * mm)
        canvas.setFont(self._font_name(), 8)
        canvas.setFillColor(colors.HexColor("#647780"))
        canvas.drawString(22 * mm, 10.5 * mm, "AgentX Research Report")
        canvas.drawRightString(width - 22 * mm, 10.5 * mm, f"Page {document.page}")
        canvas.restoreState()

