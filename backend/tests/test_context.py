from backend.app.services.context import retrieve_chunks, build_context_prompt
from unittest.mock import patch

@patch("backend.app.services.context.get_embedding")
@patch("backend.app.services.context.query_collection")
def test_retrieve_chunks(mock_query_col, mock_get_emb):
    mock_get_emb.return_value = [0.1, 0.2]
    mock_query_col.return_value = {
        "documents": [["Python chunk 1", "Python chunk 2"]],
        "metadatas": [[{"source": "doc1.txt"}, {"source": "doc2.txt"}]],
        "ids": [["id1", "id2"]],
        "distances": [[0.12, 0.45]]
    }
    
    chunks = retrieve_chunks("What is Python?", "test_col", n_results=2)
    
    assert len(chunks) == 2
    assert chunks[0]["text"] == "Python chunk 1"
    assert chunks[0]["metadata"]["source"] == "doc1.txt"
    assert chunks[0]["id"] == "id1"
    assert chunks[0]["distance"] == 0.12
    
    # Assert calling mechanics
    mock_get_emb.assert_called_once_with("What is Python?")
    mock_query_col.assert_called_once_with(
        collection_name="test_col",
        query_embeddings=[[0.1, 0.2]],
        n_results=2
    )

def test_build_context_prompt():
    chunks = [
        {"id": "c1", "text": "This is test chunk one.", "metadata": {"source": "a.txt"}},
        {"id": "c2", "text": "This is test chunk two.", "metadata": {"source": "b.txt"}}
    ]
    
    context = build_context_prompt(chunks)
    assert '<source id="c1" file="a.txt">' in context
    assert 'This is test chunk one.' in context
    assert '<source id="c2" file="b.txt">' in context
    assert 'This is test chunk two.' in context

def test_build_context_prompt_empty():
    assert build_context_prompt([]) == "No context available."
