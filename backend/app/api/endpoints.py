import os
import shutil
import uuid
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

from backend.app.services.ingestion import extract_text
from backend.app.services.chunking import RecursiveCharacterTextSplitter
from backend.app.services.embedding import get_embeddings
from backend.app.services.vector_store import add_documents, delete_collection
from backend.app.services.context import retrieve_chunks, build_context_prompt
from backend.app.services.gemini import generate_grounded_answer

router = APIRouter()

# Default collection name for ChromaDB
COLLECTION_NAME = "documents_collection"

# Ensure upload directory exists
UPLOAD_DIR = os.path.join("data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class AskRequest(BaseModel):
    question: str
    n_results: int = 4

class AskResponse(BaseModel):
    answer: str
    sources: List[dict]

@router.post("/upload", response_model=dict)
async def upload_document(file: UploadFile = File(...)):
    """
    Ingests, chunks, embeds, and indexes an uploaded document.
    Supported extensions: .txt, .pdf, .docx, .csv
    """
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".txt", ".pdf", ".docx", ".csv"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        
    # Save the file locally
    temp_file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    try:
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # 1. Ingestion: Extract raw text
        text_content = extract_text(temp_file_path)
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="The uploaded document contains no readable text.")
            
        # 2. Chunking: Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text_content)
        
        # 3. Embedding & Vector Database: Store
        # Generate unique ids and metadata for each chunk
        ids = [f"{filename}_{idx}" for idx in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_idx": idx} for idx in range(len(chunks))]
        
        # Call Google API to get embedding vectors for all chunks
        embeddings = get_embeddings(chunks)
        
        # Add to ChromaDB index
        add_documents(
            collection_name=COLLECTION_NAME,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return {
            "message": "File processed successfully",
            "filename": filename,
            "chunks_count": len(chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """
    Answers a user question grounded on the ingested document contexts.
    """
    try:
        # 1. Retrieve matching chunks from vector store
        chunks = retrieve_chunks(req.question, COLLECTION_NAME, n_results=req.n_results)
        
        # 2. Context Building: Format chunks
        context_str = build_context_prompt(chunks)
        
        # 3. Gemini Generation: Answer grounded on context
        answer = generate_grounded_answer(req.question, context_str)
        
        # Structure references to return to client
        sources = [
            {
                "id": c["id"],
                "file": c["metadata"].get("source", "Unknown"),
                "text": c["text"]
            } for c in chunks
        ]
        
        return AskResponse(answer=answer, sources=sources)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

@router.post("/clear", response_model=dict)
async def clear_collection():
    """
    Clears all documents from the vector database collection.
    """
    try:
        delete_collection(COLLECTION_NAME)
        return {"message": "All documents cleared from the vector store."}
    except Exception as e:
        import traceback
        traceback.print_exc()
            
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )
