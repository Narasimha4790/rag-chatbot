import pytest
from unittest.mock import patch, MagicMock
from backend.app.services.embedding import get_embedding, get_embeddings

@patch("backend.app.services.embedding.genai.embed_content")
@patch("backend.app.services.embedding.settings")
def test_get_embedding_mock(mock_settings, mock_embed):
    mock_settings.GEMINI_API_KEY = "dummy_key"
    mock_settings.EMBEDDING_MODEL = "models/gemini-embedding-001"
    mock_embed.return_value = {"embedding": [0.1, 0.2, 0.3]}
    
    emb = get_embedding("Hello test text")
    assert emb == [0.1, 0.2, 0.3]
    mock_embed.assert_called_once_with(
        model="models/gemini-embedding-001",
        content="Hello test text",
        task_type="retrieval_document",
        output_dimensionality=768
    )

@patch("backend.app.services.embedding.genai.embed_content")
@patch("backend.app.services.embedding.settings")
def test_get_embeddings_mock(mock_settings, mock_embed):
    mock_settings.GEMINI_API_KEY = "dummy_key"
    mock_settings.EMBEDDING_MODEL = "models/gemini-embedding-001"
    mock_embed.return_value = {"embedding": [[0.1, 0.2], [0.3, 0.4]]}
    
    embs = get_embeddings(["Text one", "Text two"])
    assert embs == [[0.1, 0.2], [0.3, 0.4]]
    mock_embed.assert_called_once_with(
        model="models/gemini-embedding-001",
        content=["Text one", "Text two"],
        task_type="retrieval_document",
        output_dimensionality=768
    )

def test_missing_api_key_error():
    with patch("backend.app.services.embedding.settings") as mock_settings:
        mock_settings.GEMINI_API_KEY = ""
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            get_embedding("hello")
