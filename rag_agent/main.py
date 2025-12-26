"""
Main FastAPI application for the RAG Agent system
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import logging
import sys
import os

# Load environment variables
load_dotenv()

# Add the backend directory to the path so we can import from rag_agent
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from rag_agent.api.agent_endpoints import router as agent_router
from rag_agent.config import Config


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="OpenRouter Agent-powered RAG system for AI Robotics textbook",
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
app.include_router(agent_router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "RAG Agent API",
        "version": "1.0.0",
        "endpoints": [
            "/api/v1/chat",
            "/api/v1/session",
            "/api/v1/health"
        ]
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "rag_agent.main:app",
        host=Config.FASTAPI_HOST,
        port=Config.FASTAPI_PORT,
        reload=Config.FASTAPI_DEBUG
    )