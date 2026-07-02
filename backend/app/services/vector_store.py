import chromadb
from typing import List, Dict, Any
from backend.app.core.config import settings

# Lazy initialization of the HTTP Client to avoid connection side-effects on import
_chroma_client = None

def get_client() -> chromadb.HttpClient:
    """Returns the initialized ChromaDB HTTP client singleton."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT
        )
    return _chroma_client

def get_or_create_collection(collection_name: str):
    """Retrieves an existing ChromaDB collection or creates a new one."""
    return get_client().get_or_create_collection(name=collection_name)

def add_documents(
    collection_name: str,
    documents: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict[str, Any]],
    ids: List[str]
):
    """Stores text chunks, custom metadata, and their corresponding embedding vectors in ChromaDB."""
    collection = get_or_create_collection(collection_name)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

def query_collection(
    collection_name: str,
    query_embeddings: List[List[float]],
    n_results: int = 4
) -> Dict[str, Any]:
    """Retrieves top-K closest matching documents based on the supplied query embeddings."""
    collection = get_or_create_collection(collection_name)
    return collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results
    )

def delete_collection(collection_name: str):
    """Clears a collection from ChromaDB."""
    try:
        get_client().delete_collection(name=collection_name)
    except Exception:
        pass
