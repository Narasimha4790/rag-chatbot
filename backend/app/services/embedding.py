import google.generativeai as genai
from typing import List
from backend.app.core.config import settings

# Configure Gemini SDK
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)

def get_embedding(text: str, model: str = settings.EMBEDDING_MODEL) -> List[float]:
    """Generates an embedding vector for a single text chunk."""
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured. Please set it in your environment.")
        
    clean_text = text.strip().replace("\n", " ")
    if not clean_text:
        return [0.0] * 768
        
    response = genai.embed_content(
        model=model,
        content=clean_text,
        task_type="retrieval_document",
        output_dimensionality=768
    )
    return response["embedding"]

def get_embeddings(texts: List[str], model: str = settings.EMBEDDING_MODEL) -> List[List[float]]:
    """Generates embedding vectors for a list of text chunks in a batch."""
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured. Please set it in your environment.")
        
    clean_texts = [t.strip().replace("\n", " ") for t in texts]
    if not clean_texts:
        return []
        
    response = genai.embed_content(
        model=model,
        content=clean_texts,
        task_type="retrieval_document",
        output_dimensionality=768
    )
    return response["embedding"]
