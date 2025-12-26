"""
Configuration module for RAG Pipeline
Handles API keys, service endpoints, and other configuration parameters
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class containing all settings for the RAG pipeline"""

    # Cohere API Configuration
    COHERE_API_KEY = os.getenv('COHERE_API_KEY')
    COHERE_MODEL = os.getenv('COHERE_MODEL', 'embed-multilingual-v3.0')

    # Qdrant Configuration
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'book_content')

    # Website URL for content extraction
    WEBSITE_URL = os.getenv('WEBSITE_URL', 'https://your-book-site.github.io')

    # Chunking parameters
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '512'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '64'))

    # Rate limiting parameters
    COHERE_RATE_LIMIT_DELAY = float(os.getenv('COHERE_RATE_LIMIT_DELAY', '1.0'))

    # Validation
    @classmethod
    def validate(cls):
        """Validate that required configuration parameters are set"""
        required_vars = [
            'COHERE_API_KEY',
            'QDRANT_URL',
            'QDRANT_API_KEY'
        ]

        missing_vars = [var for var in required_vars if not getattr(cls, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")