"""Vector database loading and management."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, OLLAMA_BASE_URL, PERSIST_DIRECTORY
from src.utils.logging import log_error


def load_vector_store() -> Chroma:
    """
    Load the Chroma vector database.

    Raises:
        FileNotFoundError: If the vector database doesn't exist

    Returns:
        Initialized Chroma vector store
    """
    if not PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Vector DB not found at {PERSIST_DIRECTORY}. Run scripts/build_kb.py first."
        )

    embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    embedding_model.embed_query("warmup")
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIRECTORY),
        embedding_function=embedding_model,
    )


def load_all_documents(vector_db: Chroma) -> list[Document]:
    """
    Retrieve all documents from the vector database.

    Args:
        vector_db: Chroma vector store instance

    Returns:
        List of all stored documents

    Raises:
        ValueError: If the vector database is empty
    """
    data = vector_db.get()
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    if not documents:
        raise ValueError("Vector DB is empty. Run scripts/build_kb.py again to rebuild the knowledge base.")

    loaded_documents = []
    for page_content, metadata in zip(documents, metadatas):
        loaded_documents.append(Document(page_content=page_content, metadata=metadata or {}))
    return loaded_documents

