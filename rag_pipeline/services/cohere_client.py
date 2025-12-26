"""
Cohere client configuration and service
Handles connection and operations with Cohere API for embedding generation
"""
import cohere
from typing import List, Optional
from ..config import Config
from ..utils.logging_config import get_logger
from ..utils.exceptions import EmbeddingGenerationError
from ..utils.helpers import retry_with_backoff


class CohereService:
    """
    Service class for interacting with Cohere API
    """
    def __init__(self):
        self.logger = get_logger()
        self.config = Config
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Cohere client with configuration"""
        try:
            if not self.config.COHERE_API_KEY:
                raise EmbeddingGenerationError("Cohere API key is not configured")

            self.client = cohere.Client(self.config.COHERE_API_KEY)
            self.logger.info("Cohere client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Cohere client: {str(e)}")
            raise EmbeddingGenerationError(f"Failed to initialize Cohere client: {str(e)}")

    def generate_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Cohere API

        Args:
            texts: List of texts to generate embeddings for
            model: Model to use for embeddings (optional, defaults to config)

        Returns:
            List of embedding vectors
        """
        if not texts:
            self.logger.warning("No texts provided for embedding generation")
            return []

        # Use the configured model or the provided one
        embedding_model = model or self.config.COHERE_MODEL

        try:
            # Use retry_with_backoff to handle rate limits and temporary failures
            def _generate_embeddings():
                response = self.client.embed(
                    texts=texts,
                    model=embedding_model,
                    input_type="search_document"  # Appropriate for document content
                )
                return response.embeddings

            embeddings = retry_with_backoff(
                _generate_embeddings,
                max_retries=3,
                backoff_factor=self.config.COHERE_RATE_LIMIT_DELAY
            )

            self.logger.info(f"Generated embeddings for {len(texts)} texts using model {embedding_model}")
            return embeddings

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {str(e)}")
            raise EmbeddingGenerationError(f"Failed to generate embeddings: {str(e)}")

    def generate_single_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Text to generate embedding for
            model: Model to use for embedding (optional, defaults to config)

        Returns:
            Embedding vector
        """
        embeddings = self.generate_embeddings([text], model)
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        else:
            raise EmbeddingGenerationError("Failed to generate embedding for single text")

    def validate_api_access(self) -> bool:
        """
        Validate that the API key has access to Cohere services

        Returns:
            True if API access is valid, False otherwise
        """
        try:
            # Try to generate a simple embedding to test API access
            test_embedding = self.generate_single_embedding("test")
            return len(test_embedding) > 0
        except Exception as e:
            self.logger.error(f"API access validation failed: {str(e)}")
            return False

    def get_model_info(self, model: Optional[str] = None) -> dict:
        """
        Get information about the embedding model

        Args:
            model: Model name to get info for (optional, defaults to config)

        Returns:
            Model information
        """
        try:
            # For now, we'll return basic info based on model name
            # In a real implementation, this would call Cohere's model info API
            embedding_model = model or self.config.COHERE_MODEL
            return {
                "model": embedding_model,
                "dimensions": 1024 if "multilingual" in embedding_model.lower() else 768,
                "type": "embeddings"
            }
        except Exception as e:
            self.logger.error(f"Failed to get model info: {str(e)}")
            raise EmbeddingGenerationError(f"Failed to get model info: {str(e)}")