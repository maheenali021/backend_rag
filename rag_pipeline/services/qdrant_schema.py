"""
Qdrant schema service for RAG Pipeline
Defines and manages the Qdrant collection schema for book content
"""
from typing import Dict, Any, Optional
from ..utils.logging_config import get_logger
from ..utils.exceptions import VectorStorageError
from ..services.qdrant_client import QdrantService
from ..config import Config


class QdrantSchemaManager:
    """
    Service class for managing Qdrant collection schema
    """
    def __init__(self):
        self.logger = get_logger()
        self.config = Config
        self.qdrant_service = QdrantService()

    def create_book_content_schema(self, vector_size: int = 1024):
        """
        Create the schema for storing book content embeddings in Qdrant

        Args:
            vector_size: Size of the embedding vectors (default 1024 for Cohere multilingual model)
        """
        try:
            self.qdrant_service.create_collection(vector_size=vector_size)
            self.logger.info(f"Created Qdrant collection schema for book content with vector size {vector_size}")
        except Exception as e:
            self.logger.error(f"Failed to create Qdrant schema: {str(e)}")
            raise VectorStorageError(f"Failed to create Qdrant schema: {str(e)}")

    def get_default_payload_schema(self) -> Dict[str, Any]:
        """
        Get the default payload schema for book content

        Returns:
            Dictionary representing the expected payload schema
        """
        return {
            "chunk_id": "keyword",  # ID of the content chunk
            "content": "text",  # The actual content text
            "source_url": "keyword",  # URL where content was found
            "chapter": "keyword",  # Chapter name/number
            "section": "keyword",  # Section name/number
            "book_title": "text",  # Title of the book
            "created_at": "datetime",  # When the content was indexed
            "model": "keyword",  # Model used for embedding
            "word_count": "integer",  # Number of words in the chunk
            "char_count": "integer",  # Number of characters in the chunk
        }

    def validate_payload(self, payload: Dict[str, Any]) -> bool:
        """
        Validate that a payload conforms to the expected schema

        Args:
            payload: Payload to validate

        Returns:
            True if payload is valid, False otherwise
        """
        required_fields = ["chunk_id", "content", "source_url"]

        for field in required_fields:
            if field not in payload:
                self.logger.error(f"Missing required field in payload: {field}")
                return False

        # Validate field types where possible
        if not isinstance(payload.get("chunk_id"), str):
            self.logger.error("chunk_id must be a string")
            return False

        if not isinstance(payload.get("content"), str):
            self.logger.error("content must be a string")
            return False

        if not isinstance(payload.get("source_url"), str):
            self.logger.error("source_url must be a string")
            return False

        return True

    def get_collection_info(self):
        """
        Get information about the current collection
        """
        try:
            return self.qdrant_service.get_collection_info()
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {str(e)}")
            raise VectorStorageError(f"Failed to get collection info: {str(e)}")

    def recreate_collection(self, vector_size: int = 1024, confirm: bool = False):
        """
        Recreate the collection (useful for development/testing)

        Args:
            vector_size: Size of the embedding vectors
            confirm: Must be True to actually perform the deletion
        """
        if not confirm:
            raise VectorStorageError("Recreation requires explicit confirmation (confirm=True)")

        try:
            # Delete existing collection
            self.qdrant_service.delete_collection()
            self.logger.info("Deleted existing collection")

            # Create new collection with schema
            self.create_book_content_schema(vector_size)
            self.logger.info("Recreated collection with new schema")

        except Exception as e:
            self.logger.error(f"Failed to recreate collection: {str(e)}")
            raise VectorStorageError(f"Failed to recreate collection: {str(e)}")

    def get_expected_vector_size(self, model_name: Optional[str] = None) -> int:
        """
        Get the expected vector size based on the embedding model

        Args:
            model_name: Name of the embedding model (optional, defaults to config)

        Returns:
            Expected vector size
        """
        model = model_name or self.config.COHERE_MODEL

        # Common Cohere model vector sizes
        model_sizes = {
            "embed-multilingual-v3.0": 1024,
            "embed-english-v3.0": 1024,
            "embed-multilingual-v2.0": 768,
            "embed-english-v2.0": 4096,
        }

        # Default to 1024 if model is unknown
        return model_sizes.get(model, 1024)

    def update_collection_config(self, **kwargs):
        """
        Update collection configuration with additional parameters

        Args:
            **kwargs: Collection configuration parameters
        """
        # This would typically involve updating collection parameters
        # For now, we'll just log that this is an available capability
        self.logger.info(f"Collection configuration update requested with: {kwargs}")
        # In a real implementation, this would call Qdrant client methods to update the collection