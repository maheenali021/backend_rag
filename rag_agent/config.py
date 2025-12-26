"""
Configuration module for RAG Agent
Handles API keys, service endpoints, and other configuration parameters
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env files in multiple locations
def _load_environment_variables():
    """Load environment variables from multiple .env files"""
    import sys
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(current_file_dir)
    project_root = os.path.dirname(backend_dir)
    pipeline_dir = os.path.join(backend_dir, 'rag_pipeline')

    # Load .env files in order (only set variables that aren't already set)
    env_files = [
        os.path.join(project_root, '.env'),  # Project root .env - lowest priority
        os.path.join(pipeline_dir, '.env'),  # rag_pipeline/.env - medium priority
        os.path.join(backend_dir, 'rag_agent', '.env')  # rag_agent/.env - highest priority
    ]

    for env_file in env_files:
        if os.path.isfile(env_file):
            load_dotenv(env_file, override=False)  # Only set if not already set

# Load environment variables when module is imported
_load_environment_variables()

class Config:
    """Configuration class containing all settings for the RAG agent"""

    # OpenRouter API Configuration
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'mistralai/mistral-7b-instruct')

    # Agent Configuration
    AGENT_NAME = os.getenv('AGENT_NAME', 'AI-Textbook-Agent')
    AGENT_INSTRUCTIONS = os.getenv('AGENT_INSTRUCTIONS',
        'You are an AI assistant that answers questions based on content from the AI Robotics textbook. '
        'Always ground your responses in the retrieved content and provide source attribution. '
        'Never hallucinate information.')

    # RAG Pipeline Configuration
    RETRIEVAL_API_URL = os.getenv('RETRIEVAL_API_URL', 'http://localhost:8000/api/v1/retrieval/validate')
    RETRIEVAL_TIMEOUT = int(os.getenv('RETRIEVAL_TIMEOUT', '30'))

    # FastAPI Configuration
    FASTAPI_HOST = os.getenv('FASTAPI_HOST', '0.0.0.0')
    FASTAPI_PORT = int(os.getenv('FASTAPI_PORT', '8001'))
    FASTAPI_DEBUG = os.getenv('FASTAPI_DEBUG', 'True').lower() == 'true'

    # Vector Database Configuration (for direct access if needed)
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'book_content')

    # Cohere Configuration (for embedding generation)
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    COHERE_MODEL = os.getenv('COHERE_MODEL', 'embed-multilingual-v3.0')

    # Validation thresholds
    MIN_CONFIDENCE_SCORE = float(os.getenv('MIN_CONFIDENCE_SCORE', '0.3'))
    MIN_SIMILARITY_SCORE = float(os.getenv('MIN_SIMILARITY_SCORE', '0.5'))
    MAX_RESPONSE_TOKENS = int(os.getenv('MAX_RESPONSE_TOKENS', '1000'))

    # Validation
    @classmethod
    def validate(cls):
        """Validate that required configuration parameters are set"""
        required_vars = [
            'OPENROUTER_API_KEY',
            'QDRANT_URL',
            'QDRANT_API_KEY'
        ]

        # Check if we have at least one of the LLM API keys
        openrouter_key = getattr(cls, 'OPENROUTER_API_KEY', None)
        gemini_key = getattr(cls, 'GEMINI_API_KEY', None)
        llm_key_present = openrouter_key or gemini_key
        if not llm_key_present:
            raise ValueError("Either OPENROUTER_API_KEY or GEMINI_API_KEY must be set")

        # Check if we have the Cohere API key for embeddings
        cohere_key = getattr(cls, 'COHERE_API_KEY', None)
        if not cohere_key:
            raise ValueError("COHERE_API_KEY must be set for embeddings")

        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")