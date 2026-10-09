"""Build the vector database from PDF using pymupdf4llm for Markdown extraction."""

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


def _resolve_pdf_path(pdf_path: Path) -> Path:
    """Resolve a file path, or the single PDF in the default KB_PDF folder."""
    pdf_path = Path(pdf_path)
    if pdf_path.is_dir():
        candidates = sorted(pdf_path.glob("*.pdf"))
        if not candidates:
            raise FileNotFoundError(f"No PDF file found in: {pdf_path}")
        return candidates[0]
    return pdf_path


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


def build_vector_store(
    pdf_path: Path = PDF_PATH,
    persist_dir: Path = PERSIST_DIRECTORY,
    chunk_size: int = PDF_CHUNK_SIZE,
    chunk_overlap: int = PDF_CHUNK_OVERLAP,
    embedding_model: Any = None,
) -> list[Document]:
    """
    Build and persist the vector database from PDF.

    Uses pymupdf4llm for Markdown-aware extraction, MarkdownTextSplitter
    for structure-preserving chunking, OllamaEmbeddings for vectors,
    and Chroma for persistence.

    Args:
        pdf_path: Path to the source PDF.
        persist_dir: Directory where the Chroma database will be persisted.
        chunk_size: Maximum character length for each document chunk.
        chunk_overlap: Number of characters to overlap between chunks.
        embedding_model: Optional embedding model instance (defaults to OllamaEmbeddings).

    Returns:
        List of created Document chunks.
    """
    pdf_path = _resolve_pdf_path(pdf_path)
    persist_dir = Path(persist_dir)
    persist_dir.parent.mkdir(parents=True, exist_ok=True)
    documents = load_pdf_documents(pdf_path=pdf_path)

    # MarkdownTextSplitter respects Markdown heading boundaries,
    # producing chunks that don't split mid-section.
    text_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} pages -> created {len(chunks)} chunks.")

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
