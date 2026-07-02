import pytest
from unittest.mock import patch, MagicMock
from backend.app.services.gemini import generate_grounded_answer

@patch("backend.app.services.gemini.genai.GenerativeModel")
@patch("backend.app.services.gemini.settings")
def test_generate_grounded_answer_success(mock_settings, mock_gen_model):
    mock_settings.GEMINI_API_KEY = "dummy_api_key"
    mock_settings.LLM_MODEL = "models/gemini-2.5-flash"
    
    mock_model_inst = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is a grounded answer."
    mock_model_inst.generate_content.return_value = mock_response
    mock_gen_model.return_value = mock_model_inst
    
    ans = generate_grounded_answer("test question?", "<context>mock context</context>")
    assert ans == "This is a grounded answer."
    
    mock_gen_model.assert_called_once()
    mock_model_inst.generate_content.assert_called_once()

def test_generate_grounded_answer_missing_key():
    with patch("backend.app.services.gemini.settings") as mock_settings:
        mock_settings.GEMINI_API_KEY = ""
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            generate_grounded_answer("test?", "context")
