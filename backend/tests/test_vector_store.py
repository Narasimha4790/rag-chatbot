import pytest
from unittest.mock import patch, MagicMock
from backend.app.services.vector_store import (
    get_or_create_collection,
    add_documents,
    query_collection,
    delete_collection
)

@pytest.fixture(autouse=True)
def mock_get_client():
    """Mocks the get_client function in the vector_store module."""
    mock_client = MagicMock()
    with patch("backend.app.services.vector_store.get_client", return_value=mock_client):
        yield mock_client

def test_get_or_create_collection(mock_get_client):
    mock_collection = MagicMock()
    mock_get_client.get_or_create_collection.return_value = mock_collection
    
    col = get_or_create_collection("test_col")
    assert col == mock_collection
    mock_get_client.get_or_create_collection.assert_called_once_with(name="test_col")

def test_add_documents(mock_get_client):
    mock_collection = MagicMock()
    mock_get_client.get_or_create_collection.return_value = mock_collection
    
    docs = ["doc1", "doc2"]
    embs = [[0.1, 0.2], [0.3, 0.4]]
    metas = [{"source": "a"}, {"source": "b"}]
    ids = ["id1", "id2"]
    
    add_documents("test_col", docs, embs, metas, ids)
    mock_collection.add.assert_called_once_with(
        documents=docs,
        embeddings=embs,
        metadatas=metas,
        ids=ids
    )

def test_query_collection(mock_get_client):
    mock_collection = MagicMock()
    mock_get_client.get_or_create_collection.return_value = mock_collection
    mock_collection.query.return_value = {"documents": [["doc1"]]}
    
    res = query_collection("test_col", [[0.1, 0.2]], n_results=1)
    assert res == {"documents": [["doc1"]]}
    mock_collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2]],
        n_results=1
    )

def test_delete_collection(mock_get_client):
    delete_collection("test_col")
    mock_get_client.delete_collection.assert_called_once_with(name="test_col")
