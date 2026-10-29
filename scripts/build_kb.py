"""Build the vector database from PDF and DOCX knowledge-base sources."""

import shutil
from pathlib import Path
from typing import Any

import pymupdf4llm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownTextSplitter

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    PDF_CHUNK_OVERLAP,
    PDF_CHUNK_SIZE,
    PDF_PATH,
    PERSIST_DIRECTORY,
)

SUPPORTED_SOURCE_SUFFIXES = {".pdf", ".docx"}


def _resolve_source_paths(source_path: Path) -> list[Path]:
    """Resolve one source file or all supported files in a knowledge-base folder."""
    source_path = Path(source_path)
    if source_path.is_dir():
        candidates = sorted(
            path for path in source_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES
        )
        if not candidates:
            supported = ", ".join(sorted(SUPPORTED_SOURCE_SUFFIXES))
            raise FileNotFoundError(f"No supported knowledge-base files ({supported}) found in: {source_path}")
        return candidates
    if not source_path.exists():
        raise FileNotFoundError(f"Knowledge-base source not found at: {source_path}")
    if source_path.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
        raise ValueError(
            f"Unsupported knowledge-base format: {source_path.suffix}. "
            "Only .pdf and .docx files are supported."
        )
    return [source_path]


def _resolve_pdf_path(pdf_path: Path) -> Path:
    """Backward-compatible resolver for callers that explicitly provide a PDF."""
    candidates = [path for path in _resolve_source_paths(pdf_path) if path.suffix.lower() == ".pdf"]
    if not candidates:
        raise ValueError(f"Expected a PDF source, received: {pdf_path}")
    return candidates[0]


def load_pdf_documents(pdf_path: Path = PDF_PATH) -> list[Document]:
    """
    Extract text from PDF as Markdown using pymupdf4llm.

    pymupdf4llm converts each page into structured Markdown,
    preserving headings, tables, lists, and code blocks — which
    results in semantically cleaner chunks compared to raw text.

    Args:
        pdf_path: Path to the target PDF file.

    Returns:
        List of LangChain Documents, one per PDF page.

    Raises:
        FileNotFoundError: If PDF file doesn't exist.
        ValueError: If no content could be extracted from PDF.
    """
    pdf_path = _resolve_pdf_path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found at: {pdf_path}. "
            "Add your PDF content and run again."
        )

    # pymupdf4llm.to_markdown returns a list of dicts, one per page:
    # [{"metadata": {"page": 0, "file_path": "..."}, "text": "<markdown>"}]
    try:
        pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    except Exception:
        # Some valid PDFs contain layouts pymupdf4llm cannot parse. Fall back
        # to the native text layer so a usable document is still searchable.
        import pymupdf

        with pymupdf.open(str(pdf_path)) as pdf:
            pages = [
                {"metadata": {"page": index}, "text": page.get_text()}
                for index, page in enumerate(pdf)
            ]

    if not pages:
        raise ValueError(f"No content could be extracted from {pdf_path}")

    documents: list[Document] = []
    for page in pages:
        md_text: str = page.get("text", "").strip()
        if not md_text:
            continue

        # page index in pymupdf4llm is 0-based; convert to 1-based for metadata
        page_num: int = page.get("metadata", {}).get("page", 0) + 1

        documents.append(
            Document(
                page_content=md_text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_num,
                },
            )
        )

    if not documents:
        raise ValueError(f"All pages were empty in {pdf_path}")

    return documents


def load_docx_documents(docx_path: Path) -> list[Document]:
    """Extract paragraphs, headings, lists, and tables from a Word document."""
    from src.admin.parsers import parse_docx

    docx_path = Path(docx_path)
    if docx_path.suffix.lower() != ".docx":
        raise ValueError(f"Expected a .docx source, received: {docx_path}")
    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX file not found at: {docx_path}")

    documents = parse_docx(docx_path)
    for document in documents:
        document.metadata.update({"source": str(docx_path), "file_type": "docx"})
    return documents


def load_source_documents(source_path: Path = PDF_PATH) -> list[Document]:
    """Extract all PDF and DOCX files from a file or the KB_PDF directory."""
    documents: list[Document] = []
    for path in _resolve_source_paths(source_path):
        if path.suffix.lower() == ".pdf":
            source_documents = load_pdf_documents(path)
        else:
            source_documents = load_docx_documents(path)
        for document in source_documents:
            document.metadata.setdefault("source", str(path))
            document.metadata["file_name"] = path.name
            document.metadata["file_type"] = path.suffix.lower().lstrip(".")
        documents.extend(source_documents)
    return documents


def build_vector_store(
    pdf_path: Path = PDF_PATH,
    persist_dir: Path = PERSIST_DIRECTORY,
    chunk_size: int = PDF_CHUNK_SIZE,
    chunk_overlap: int = PDF_CHUNK_OVERLAP,
    embedding_model: Any = None,
) -> list[Document]:
    """
    Build and persist the vector database from all PDF and DOCX files in the source.

    Uses pymupdf4llm for Markdown-aware extraction, MarkdownTextSplitter
    for structure-preserving chunking, OllamaEmbeddings for vectors,
    and Chroma for persistence.

    Args:
        pdf_path: Path to a source file or a directory containing PDF/DOCX files.
        persist_dir: Directory where the Chroma database will be persisted.
        chunk_size: Maximum character length for each document chunk.
        chunk_overlap: Number of characters to overlap between chunks.
        embedding_model: Optional embedding model instance (defaults to OllamaEmbeddings).

    Returns:
        List of created Document chunks.
    """
    persist_dir = Path(persist_dir)
    persist_dir.parent.mkdir(parents=True, exist_ok=True)
    documents = load_source_documents(source_path=pdf_path)

    # MarkdownTextSplitter respects Markdown heading boundaries,
    # producing chunks that don't split mid-section.
    text_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    source_names = sorted({document.metadata.get("file_name", "source") for document in documents})
    print(f"Loaded {len(documents)} source pages/sections from {', '.join(source_names)} -> created {len(chunks)} chunks.")

    if embedding_model is None:
        embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
    )

    print(
        f"[OK] Vector DB saved to {persist_dir.resolve()} "
        f"in collection '{COLLECTION_NAME}'."
    )
    return chunks


if __name__ == "__main__":
    build_vector_store()
