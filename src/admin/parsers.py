"""Multi-format document parsing and metadata-enriched chunking engine.

Supports:
- PDF (.pdf) via PyMuPDF4LLM and PyMuPDF
- Word Documents (.docx) via python-docx (preserving headings, tables, lists)
- School Notices and Announcements (Markdown/Text)
- FAQ Q&A pairs
- Approved Web Content (Markdown/HTML text extraction)
"""

from html.parser import HTMLParser
import re
from pathlib import Path
from typing import Any
import urllib.request

import pymupdf
import pymupdf4llm
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter

from src.admin.schemas import DocumentRecord
from src.config import PDF_CHUNK_OVERLAP, PDF_CHUNK_SIZE
from src.utils.logging import log_error, log_step


class HTMLTextExtractor(HTMLParser):
    """Simple robust HTML text extractor stripping script and style tags."""

    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []
        self._ignore = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in ("script", "style", "noscript", "svg", "head"):
            self._ignore = True
        elif tag.lower() in ("p", "div", "h1", "h2", "h3", "h4", "li", "tr"):
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in ("script", "style", "noscript", "svg", "head"):
            self._ignore = False
        elif tag.lower() in ("p", "div", "h1", "h2", "h3", "h4", "li", "tr"):
            self._pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignore:
            cleaned = data.strip()
            if cleaned:
                self._pieces.append(f"{cleaned} ")

    def get_text(self) -> str:
        text = "".join(self._pieces)
        return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def parse_pdf(file_path: Path) -> list[Document]:
    """Parse PDF file into structured Markdown documents per page."""
    if not file_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    pages = []
    try:
        pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
    except Exception:
        # Fallback to standard PyMuPDF
        with pymupdf.open(str(file_path)) as pdf:
            pages = [
                {"metadata": {"page": idx}, "text": page.get_text()}
                for idx, page in enumerate(pdf)
            ]

    docs: list[Document] = []
    for idx, page in enumerate(pages):
        text = page.get("text", "").strip()
        if not text:
            continue
        page_num = page.get("metadata", {}).get("page", idx) + 1
        docs.append(Document(page_content=text, metadata={"page": page_num}))

    if not docs:
        raise ValueError(f"No readable text content extracted from {file_path.name}")
    return docs


def parse_docx(file_path: Path) -> list[Document]:
    """
    Parse Word (.docx) document preserving headings, paragraphs, and markdown tables.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Word document not found: {file_path}")

    doc = DocxDocument(str(file_path))
    content_parts: list[str] = []

    for elem in doc.paragraphs:
        text = elem.text.strip()
        if not text:
            continue
        style_name = elem.style.name.lower() if elem.style else ""
        if "heading 1" in style_name:
            content_parts.append(f"\n# {text}\n")
        elif "heading 2" in style_name:
            content_parts.append(f"\n## {text}\n")
        elif "heading 3" in style_name:
            content_parts.append(f"\n### {text}\n")
        elif "list" in style_name:
            content_parts.append(f"- {text}")
        else:
            content_parts.append(text)

    # Process tables into Markdown tables
    for table in doc.tables:
        table_rows: list[str] = []
        headers: list[str] = []
        for r_idx, row in enumerate(table.rows):
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if r_idx == 0:
                headers = cells
                table_rows.append("| " + " | ".join(cells) + " |")
                table_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
            else:
                table_rows.append("| " + " | ".join(cells) + " |")
        if table_rows:
            content_parts.append("\n" + "\n".join(table_rows) + "\n")

    full_text = "\n\n".join(content_parts).strip()
    if not full_text:
        raise ValueError(f"No readable content found in Word file {file_path.name}")

    return [Document(page_content=full_text, metadata={"page": 1})]


def parse_notice(record: DocumentRecord) -> list[Document]:
    """Parse school circular or notice text."""
    body = (record.content_text or "").strip()
    if not body:
        raise ValueError("Notice content is empty")

    header = f"# {record.title}\n"
    if record.school_campus:
        header += f"**Campus:** {record.school_campus} | **Academic Year:** {record.academic_year}\n"
    if record.effective_date:
        header += f"**Date of Issuance:** {record.effective_date}\n"
    header += "---\n\n"

    return [Document(page_content=header + body, metadata={"page": 1})]


def parse_faq(record: DocumentRecord) -> list[Document]:
    """
    Parse FAQ question-and-answer pairs into retrieval documents.

    Supports structured Q&A formats:
    - Multiple Q&A pairs separated by blank lines or "Q:" and "A:" prefixes
    - Single question and answer pair
    """
    body = (record.content_text or "").strip()
    if not body:
        raise ValueError("FAQ content is empty")

    # Split by standard FAQ patterns if multiple pairs are present
    pairs: list[str] = []
    current_pair: list[str] = []

    for line in body.splitlines():
        trimmed = line.strip()
        if re.match(r"^(Q|Question)\s*[:\d\.\-]", trimmed, re.IGNORECASE) and current_pair:
            pairs.append("\n".join(current_pair).strip())
            current_pair = [trimmed]
        elif trimmed:
            current_pair.append(trimmed)

    if current_pair:
        pairs.append("\n".join(current_pair).strip())

    if not pairs:
        pairs = [body]

    docs: list[Document] = []
    for idx, pair in enumerate(pairs):
        entry_text = f"### FAQ: {record.title}\n{pair}"
        docs.append(Document(page_content=entry_text, metadata={"page": idx + 1}))

    return docs


def parse_web_content(record: DocumentRecord) -> list[Document]:
    """Parse approved website content from direct text or fetched URL."""
    body = (record.content_text or "").strip()

    if not body and record.source_url:
        try:
            req = urllib.request.Request(
                record.source_url,
                headers={"User-Agent": "BBHS-KnowledgeBase-Indexer/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                raw_html = resp.read().decode(charset, errors="replace")
                parser = HTMLTextExtractor()
                parser.feed(raw_html)
                body = parser.get_text()
        except Exception as exc:
            log_error("WEB_CONTENT_FETCH_FAILED", exc, url=record.source_url)
            raise ValueError(f"Failed to fetch content from {record.source_url}: {exc}") from exc

    if not body:
        raise ValueError("Web content is empty")

    header = f"# {record.title}\n"
    if record.source_url:
        header += f"**Source:** {record.source_url}\n"
    header += "---\n\n"

    return [Document(page_content=header + body, metadata={"page": 1})]


def load_raw_documents_for_record(record: DocumentRecord, kb_dir: Path) -> list[Document]:
    """Extract raw LangChain documents for any supported document record."""
    content_type = record.content_type.lower()

    if content_type == "pdf":
        if not record.file_name:
            raise ValueError(f"No file associated with PDF document {record.id}")
        return parse_pdf(kb_dir / record.file_name)

    elif content_type == "docx":
        if not record.file_name:
            raise ValueError(f"No file associated with Word document {record.id}")
        return parse_docx(kb_dir / record.file_name)

    elif content_type == "notice":
        return parse_notice(record)

    elif content_type == "faq":
        return parse_faq(record)

    elif content_type == "web_content":
        return parse_web_content(record)

    else:
        raise ValueError(f"Unsupported content type: {record.content_type}")


def chunk_and_enrich_document(
    record: DocumentRecord,
    kb_dir: Path,
    chunk_size: int = PDF_CHUNK_SIZE,
    chunk_overlap: int = PDF_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Extract, split, and enrich document chunks with full operational metadata.

    Every chunk receives:
    - doc_id: Unique document identifier for targeted deletion
    - title: Document title
    - category: Admissions, Academics, Transport, etc.
    - school_campus: Main Campus, Primary Wing, etc.
    - academic_year: Academic Year (e.g., 2026-2027)
    - language: Document language
    - version: Document version
    - status: Current approval status
    - effective_date, expiry_date: Temporal validity
    - page: Page or section number
    - source: Source display title
    """
    raw_docs = load_raw_documents_for_record(record, kb_dir)
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_chunks = splitter.split_documents(raw_docs)

    enriched_chunks: list[Document] = []
    for idx, chunk in enumerate(split_chunks):
        page_num = chunk.metadata.get("page", 1)
        enriched_metadata: dict[str, Any] = {
            "doc_id": record.id,
            "title": record.title,
            "content_type": record.content_type,
            "category": record.category,
            "school_campus": record.school_campus,
            "academic_year": record.academic_year,
            "language": record.language,
            "version": record.version,
            "status": record.status,
            "effective_date": record.effective_date or "",
            "expiry_date": record.expiry_date or "",
            "page": page_num,
            "chunk_idx": idx,
            "source": record.title,
        }
        enriched_chunks.append(
            Document(page_content=chunk.page_content, metadata=enriched_metadata)
        )

    log_step(
        "DOC_CHUNKS_READY",
        doc_id=record.id,
        content_type=record.content_type,
        chunks=len(enriched_chunks),
    )
    return enriched_chunks
