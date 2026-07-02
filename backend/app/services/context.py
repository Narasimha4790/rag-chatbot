from typing import List, Dict, Any
from backend.app.services.embedding import get_embedding
from backend.app.services.vector_store import query_collection

def retrieve_chunks(
    query: str,
    collection_name: str,
    n_results: int = 4
) -> List[Dict[str, Any]]:
    """
    Retrieves the top-K semantically relevant document chunks matching the query.
    
    1. Generates embedding vector for the user query text.
    2. Queries ChromaDB for closest matches.
    3. Normalizes results into a list of dictionaries.
    """
    # Get the embedding for the query
    query_emb = get_embedding(query)
    
    # Query vector store (wrap query_emb in a list since query_collection expects batch vectors)
    search_results = query_collection(
        collection_name=collection_name,
        query_embeddings=[query_emb],
        n_results=n_results
    )
    
    # Restructure ChromaDB outputs into clean list of dictionaries
    retrieved_items = []
    
    # Search results are batched, extract the first batch (idx 0)
    if search_results and "documents" in search_results and search_results["documents"]:
        documents = search_results["documents"][0]
        metadatas = search_results["metadatas"][0] if search_results.get("metadatas") else []
        ids = search_results["ids"][0] if search_results.get("ids") else []
        distances = search_results["distances"][0] if search_results.get("distances") else []
        
        for idx in range(len(documents)):
            retrieved_items.append({
                "id": ids[idx] if idx < len(ids) else f"chunk_{idx}",
                "text": documents[idx],
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "distance": distances[idx] if idx < len(distances) else 0.0
            })
            
    return retrieved_items

def build_context_prompt(chunks: List[Dict[str, Any]]) -> str:
    """
    Builds a structured grounded context string from retrieved chunks
    using XML-style layout to prevent instruction injection.
    """
    if not chunks:
        return "No context available."
        
    context_blocks = []
    for chunk in chunks:
        source_name = chunk["metadata"].get("source", "Unknown Source")
        doc_id = chunk.get("id", "Unknown ID")
        
        block = (
            f'<source id="{doc_id}" file="{source_name}">\n'
            f'{chunk["text"]}\n'
            f'</source>'
        )
        context_blocks.append(block)
        
    return "\n\n".join(context_blocks)
