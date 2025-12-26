"""
Main FastAPI application for the retrieval validation service
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging
import sys
import os

# Load environment variables
load_dotenv()

# Add the backend directory to the path so we can import from rag_pipeline
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from rag_pipeline.validation.api.validation_endpoints import router as validation_router
from rag_pipeline.config import Config


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Retrieval Validation API",
    description="API for validating retrieval quality in the RAG pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(validation_router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Retrieval Validation API",
        "version": "1.0.0",
        "endpoints": [
            "/api/v1/retrieval/validate",
            "/api/v1/retrieval/test-interface",
            "/api/v1/retrieval/health"
        ]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "retrieval-validation",
        "timestamp": time.time()
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "rag_pipeline.validation.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )