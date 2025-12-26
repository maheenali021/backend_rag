import os
from rag_agent.api.agent_endpoints import router as agent_router
from rag_agent.config import Config
from fastapi import FastAPI
import logging
import sys
import requests
import json

# Add the backend directory to the path so we can import from rag_agent
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Ensure logs go to console
        logging.FileHandler("app.log")      # Optional: also log to file
    ]
)
logger = logging.getLogger(__name__)
logger.info("Logging configured successfully for main app")

# Create FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="Google Gemini Agent-powered RAG system for AI Robotics textbook",
    version="1.0.0"
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(agent_router, prefix="/api/v1")

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

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "rag-agent-api"}

# For Hugging Face Space, we'll also support a Gradio interface
def query_agent(query: str) -> str:
    '''Function to query the RAG agent'''
    # Use the same logic as the API but for Gradio
    try:
        # This simulates the API call internally
        from rag_agent.services.rag_agent_service import RAGAgentService
        agent_service = RAGAgentService()
        
        from rag_agent.models.agent_models import AgentRequest
        request = AgentRequest(query=query)
        response = agent_service.process_request(request)
        
        return response.response
    except Exception as e:
        return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),  
        reload=Config.FASTAPI_DEBUG
    )
