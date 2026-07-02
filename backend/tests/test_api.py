import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.app.main import app

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

@patch("backend.app.api.endpoints.extract_text")
@patch("backend.app.api.endpoints.get_embeddings")
@patch("backend.app.api.endpoints.add_documents")
def test_upload_document(mock_add_docs, mock_get_embs, mock_extract):
    mock_extract.return_value = "This is some extracted text from a PDF document."
    mock_get_embs.return_value = [[0.1, 0.2]]
    
    # Simulate file upload payload
    file_payload = {"file": ("test.pdf", b"pdf content bytes", "application/pdf")}
    res = client.post("/api/upload", files=file_payload)
    
    assert res.status_code == 200
    data = res.json()
    assert data["message"] == "File processed successfully"
    assert data["filename"] == "test.pdf"
    assert data["chunks_count"] == 1
    
    mock_extract.assert_called_once()
    mock_get_embs.assert_called_once()
    mock_add_docs.assert_called_once()

@patch("backend.app.api.endpoints.retrieve_chunks")
@patch("backend.app.api.endpoints.generate_grounded_answer")
def test_ask_question(mock_gen_answer, mock_retrieve):
    mock_retrieve.return_value = [
        {"id": "doc.pdf_0", "text": "Python is dynamic.", "metadata": {"source": "doc.pdf"}}
    ]
    mock_gen_answer.return_value = "Python is dynamic according to the context."
    
    payload = {"question": "What is Python?", "n_results": 2}
    res = client.post("/api/ask", json=payload)
    
    assert res.status_code == 200
    data = res.json()
    assert data["answer"] == "Python is dynamic according to the context."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["file"] == "doc.pdf"
    assert data["sources"][0]["text"] == "Python is dynamic."
    
    mock_retrieve.assert_called_once_with("What is Python?", "documents_collection", n_results=2)
    mock_gen_answer.assert_called_once()

@patch("backend.app.api.endpoints.delete_collection")
def test_clear_collection(mock_delete):
    res = client.post("/api/clear")
    assert res.status_code == 200
    assert res.json()["message"] == "All documents cleared from the vector store."
    mock_delete.assert_called_once_with("documents_collection")
