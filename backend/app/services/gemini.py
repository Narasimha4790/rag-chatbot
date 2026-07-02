import google.generativeai as genai
from backend.app.core.config import settings

# Configure Gemini SDK
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. You must answer the user's question "
    "strictly using the facts provided in the context section. "
    "If the provided context does not contain the answer, or if the context "
    "is empty, you must state exactly: 'I cannot find the answer in the provided documents.' "
    "Do not extrapolate, assume, or use any outside knowledge. Be factual and direct."
)

def generate_grounded_answer(
    question: str,
    context: str,
    model_name: str = settings.LLM_MODEL
) -> str:
    """
    Constructs a structured grounded prompt, queries the Gemini model,
    and returns the grounded textual answer.
    """
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured. Please set it in your environment.")
        
    # Set system instruction to enforce strict context grounding
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_INSTRUCTION
    )
    
    prompt = (
        f"Use the following source blocks to answer the question:\n\n"
        f"<context>\n"
        f"{context}\n"
        f"</context>\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    
    response = model.generate_content(prompt)
    return response.text
