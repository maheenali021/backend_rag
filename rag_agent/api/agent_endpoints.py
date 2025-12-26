"""
Agent endpoints for the RAG Agent system
FastAPI endpoints for agent interaction and conversation management
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
import time
import logging

from ..models.agent_models import AgentRequest, AgentResponse, ConversationSession
from ..services.rag_agent_service import RAGAgentService
from ..utils.validation_utils import ResponseValidator


# Set up logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(tags=["agent"])


@router.post("/chat", response_model=AgentResponse)
async def chat_with_agent(request: AgentRequest):
    """
    Chat endpoint for interacting with the RAG agent
    Processes user queries and returns grounded responses with source attribution
    """
    logger.info(f"Received chat request: {request.query[:50]}...")
    start_time = time.time()

    try:
        # Ensure environment variables are loaded in this context
        import os
        from dotenv import load_dotenv

        # Load environment variables to ensure they're available
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        backend_dir = os.path.dirname(current_dir)
        pipeline_dir = os.path.join(backend_dir, 'rag_pipeline')
        project_root = os.path.dirname(backend_dir)

        # Load in order of precedence
        env_files = [
            os.path.join(project_root, '.env'),
            os.path.join(pipeline_dir, '.env'),
            os.path.join(current_dir, 'rag_agent', '.env')
        ]

        for env_file in env_files:
            if os.path.isfile(env_file):
                load_dotenv(env_file, override=True)  # Override to ensure fresh values

        logger.info("Environment variables loaded successfully")

        # Initialize agent service within the function to avoid import-time failures
        agent_service = RAGAgentService()
        logger.info("RAG Agent Service initialized successfully")

        # Process the request using the agent service
        response = agent_service.process_request(request)
        logger.info(f"Request processed successfully, response length: {len(response.response)}")

        # Validate the response
        validation_result = ResponseValidator.validate_agent_response(response)
        if not validation_result["is_valid"]:
            logger.warning(f"Response validation issues: {validation_result['issues']}")

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        logger.info(f"Agent response completed in {response_time_ms:.2f}ms for query: '{request.query[:50]}...' "
                   f"confidence: {response.confidence_score:.2f}, chunks: {len(response.retrieved_chunks)}")

        return response

    except Exception as e:
        logger.error(f"Error processing agent request: {str(e)}")
        # More specific error message based on common issues
        error_msg = str(e)
        if "COHERE_API_KEY" in error_msg or "environment variable" in error_msg.lower():
            error_msg = "COHERE_API_KEY environment variable is not set or invalid"
        elif "QDRANT" in error_msg.upper():
            error_msg = "Qdrant configuration issue - check URL and API key"
        elif "OPENROUTER_API_KEY" in error_msg or "OPENROUTER" in error_msg:
            error_msg = "OpenRouter API key issue - check configuration"
        logger.error(f"Agent processing failed: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {error_msg}")


@router.post("/session", response_model=ConversationSession)
async def create_conversation_session():
    """
    Create a new conversation session
    """
    try:
        # Ensure environment variables are loaded in this context
        import os
        from dotenv import load_dotenv

        # Load environment variables to ensure they're available
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        backend_dir = os.path.dirname(current_dir)
        pipeline_dir = os.path.join(backend_dir, 'rag_pipeline')
        project_root = os.path.dirname(backend_dir)

        # Load in order of precedence
        env_files = [
            os.path.join(project_root, '.env'),
            os.path.join(pipeline_dir, '.env'),
            os.path.join(current_dir, 'rag_agent', '.env')
        ]

        for env_file in env_files:
            if os.path.isfile(env_file):
                load_dotenv(env_file, override=True)  # Override to ensure fresh values

        agent_service = RAGAgentService()
        session = agent_service.create_conversation_session()
        logger.info(f"Created new conversation session: {session.id}")
        return session
    except Exception as e:
        logger.error(f"Error creating conversation session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("/session/{session_id}", response_model=ConversationSession)
async def get_conversation_session(session_id: str):
    """
    Get conversation session details and history
    """
    try:
        # Ensure environment variables are loaded in this context
        import os
        from dotenv import load_dotenv

        # Load environment variables to ensure they're available
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        backend_dir = os.path.dirname(current_dir)
        pipeline_dir = os.path.join(backend_dir, 'rag_pipeline')
        project_root = os.path.dirname(backend_dir)

        # Load in order of precedence
        env_files = [
            os.path.join(project_root, '.env'),
            os.path.join(pipeline_dir, '.env'),
            os.path.join(current_dir, 'rag_agent', '.env')
        ]

        for env_file in env_files:
            if os.path.isfile(env_file):
                load_dotenv(env_file, override=True)  # Override to ensure fresh values

        agent_service = RAGAgentService()
        session = agent_service.get_conversation_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Conversation session not found")

        logger.info(f"Retrieved conversation session: {session_id}")
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session: {str(e)}")


@router.delete("/session/{session_id}")
async def clear_conversation_session(session_id: str):
    """
    Clear a conversation session
    """
    try:
        agent_service = RAGAgentService()
        success = agent_service.clear_conversation(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation session not found")

        logger.info(f"Cleared conversation session: {session_id}")
        return {"message": "Conversation session cleared successfully", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing conversation session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the agent service
    """
    try:
        # Initialize agent service within the function to avoid import-time failures
        agent_service = RAGAgentService()

        # Check if OpenRouter client is initialized
        if not hasattr(agent_service, 'client'):
            return {
                "status": "unhealthy",
                "service": "agent-service",
                "details": "OpenRouter client not initialized",
                "timestamp": time.time()
            }

        # Check if retrieval tool is accessible
        retrieval_accessible = agent_service.validate_retrieval_connection()

        if not retrieval_accessible:
            return {
                "status": "degraded",
                "service": "agent-service",
                "details": "Retrieval API may be unavailable",
                "timestamp": time.time()
            }

        return {
            "status": "healthy",
            "service": "agent-service",
            "details": "All systems operational",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "agent-service",
            "error": str(e),
            "timestamp": time.time()
        }


@router.get("/debug/{session_id}")
async def debug_conversation_session(session_id: str):
    """
    Debug endpoint for detailed inspection of a conversation session
    """
    try:
        agent_service = RAGAgentService()
        session = agent_service.get_conversation_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Conversation session not found")

        # Return detailed session information
        debug_info = {
            "session_id": session.id,
            "created_at": session.created_at,
            "last_interaction": session.last_interaction,
            "active": session.active,
            "history_count": len(session.history),
            "metadata_keys": list(session.metadata.keys()) if session.metadata else [],
            "history_preview": [
                {
                    "turn": i + 1,
                    "user_query_length": len(interaction["user"]),
                    "agent_response_length": len(interaction["agent"]),
                    "timestamp": interaction["timestamp"]
                }
                for i, interaction in enumerate(session.history[-5:])  # Last 5 interactions
            ] if session.history else []
        }

        logger.info(f"Debug info retrieved for session: {session_id}")
        return debug_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving debug info for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve debug info: {str(e)}")


@router.post("/validate-response")
async def validate_agent_response(response: AgentResponse):
    """
    Endpoint to validate an agent response for quality and grounding
    """
    try:
        # Initialize agent service within the function to avoid import-time failures
        agent_service = RAGAgentService()

        validation_result = ResponseValidator.validate_agent_response(response)

        logger.info(f"Response validation completed for conversation {response.conversation_id}, "
                   f"valid: {validation_result['is_valid']}")

        return {
            "is_valid": validation_result["is_valid"],
            "issues": validation_result["issues"],
            "hallucination_check": validation_result["hallucination_check"],
            "grounding_score": validation_result["grounding_score"]
        }
    except Exception as e:
        logger.error(f"Error validating agent response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Response validation failed: {str(e)}")