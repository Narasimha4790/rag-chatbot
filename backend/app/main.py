from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.endpoints import router as api_router
from backend.app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-quality REST API backend for modular RAG training.",
    version="1.0.0"
)

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Streamlit runs on a different port, allow it to query
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """Service health verification check."""
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME
    }

# Register endpoint routes
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    # Execute the ASGI server using the application instance directly to avoid module path issues
    uvicorn.run(app, host="0.0.0.0", port=8000)
