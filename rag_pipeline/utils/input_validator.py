"""
Input validation utilities for RAG Pipeline
Validates inputs to pipeline functions and API endpoints
"""
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from ..models.book_content import BookContent
from ..models.content_chunk import ContentChunk
from ..utils.logging_config import get_logger
from ..utils.exceptions import ValidationError


class InputValidator:
    """
    Utility class for validating inputs to the RAG pipeline
    """
    def __init__(self):
        self.logger = get_logger()

    def validate_url(self, url: str) -> bool:
        """
        Validate that a string is a properly formatted URL

        Args:
            url: URL string to validate

        Returns:
            True if URL is valid, False otherwise
        """
        if not url or not isinstance(url, str):
            return False

        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def validate_chunk_size(self, chunk_size: int) -> bool:
        """
        Validate chunk size parameter

        Args:
            chunk_size: Size to validate

        Returns:
            True if valid, False otherwise
        """
        return isinstance(chunk_size, int) and chunk_size > 0

    def validate_overlap_size(self, overlap_size: int) -> bool:
        """
        Validate overlap size parameter

        Args:
            overlap_size: Overlap size to validate

        Returns:
            True if valid, False otherwise
        """
        return isinstance(overlap_size, int) and overlap_size >= 0

    def validate_top_k(self, top_k: int) -> bool:
        """
        Validate top_k parameter for search operations

        Args:
            top_k: Number of results to return

        Returns:
            True if valid, False otherwise
        """
        return isinstance(top_k, int) and top_k > 0 and top_k <= 100

    def validate_similarity_threshold(self, threshold: float) -> bool:
        """
        Validate similarity threshold parameter

        Args:
            threshold: Similarity threshold to validate

        Returns:
            True if valid, False otherwise
        """
        return isinstance(threshold, (int, float)) and 0 <= threshold <= 1

    def validate_content_chunk(self, chunk: ContentChunk) -> bool:
        """
        Validate a ContentChunk object

        Args:
            chunk: ContentChunk to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(chunk, ContentChunk):
            return False

        # Check required fields
        if not chunk.id or not isinstance(chunk.id, str):
            return False

        if not chunk.book_content_id or not isinstance(chunk.book_content_id, str):
            return False

        if chunk.content is None or not isinstance(chunk.content, str):
            return False

        if not isinstance(chunk.chunk_index, int) or chunk.chunk_index < 0:
            return False

        if not isinstance(chunk.word_count, int) or chunk.word_count < 0:
            return False

        if not isinstance(chunk.char_count, int) or chunk.char_count < 0:
            return False

        return True

    def validate_book_content(self, content: BookContent) -> bool:
        """
        Validate a BookContent object

        Args:
            content: BookContent to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(content, BookContent):
            return False

        # Check required fields
        if not content.id or not isinstance(content.id, str):
            return False

        if not content.url or not isinstance(content.url, str):
            return False

        if not content.title or not isinstance(content.title, str):
            return False

        if content.content is None or not isinstance(content.content, str):
            return False

        return self.validate_url(content.url)

    def validate_pipeline_parameters(self, params: Dict[str, Any]) -> List[str]:
        """
        Validate a dictionary of pipeline parameters

        Args:
            params: Dictionary of parameters to validate

        Returns:
            List of validation error messages
        """
        errors = []

        # Validate chunk_size if present
        if 'chunk_size' in params:
            if not self.validate_chunk_size(params['chunk_size']):
                errors.append(f"Invalid chunk_size: {params['chunk_size']}")

        # Validate overlap_size if present
        if 'overlap_size' in params:
            if not self.validate_overlap_size(params['overlap_size']):
                errors.append(f"Invalid overlap_size: {params['overlap_size']}")

        # Validate top_k if present
        if 'top_k' in params:
            if not self.validate_top_k(params['top_k']):
                errors.append(f"Invalid top_k: {params['top_k']}")

        # Validate similarity threshold if present
        if 'min_similarity' in params:
            if not self.validate_similarity_threshold(params['min_similarity']):
                errors.append(f"Invalid min_similarity: {params['min_similarity']}")

        # Validate URL if present
        if 'url' in params:
            if not self.validate_url(params['url']):
                errors.append(f"Invalid URL: {params['url']}")

        return errors

    def validate_search_query(self, query: str) -> bool:
        """
        Validate a search query string

        Args:
            query: Query string to validate

        Returns:
            True if valid, False otherwise
        """
        if not query or not isinstance(query, str):
            return False

        # Check if query is not just whitespace
        if not query.strip():
            return False

        # Check length (not too short or excessively long)
        if len(query.strip()) < 2 or len(query) > 1000:
            return False

        return True

    def validate_embedding_vector(self, vector: List[float], expected_dimension: Optional[int] = None) -> bool:
        """
        Validate an embedding vector

        Args:
            vector: Vector to validate
            expected_dimension: Expected dimension of the vector (optional)

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(vector, list):
            return False

        if not vector:  # Empty vector is invalid
            return False

        if expected_dimension and len(vector) != expected_dimension:
            return False

        # Check that all elements are numbers
        for item in vector:
            if not isinstance(item, (int, float)):
                return False

        return True

    def validate_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """
        Validate metadata dictionary

        Args:
            metadata: Metadata dictionary to validate

        Returns:
            List of validation error messages
        """
        errors = []

        if not isinstance(metadata, dict):
            errors.append("Metadata must be a dictionary")
            return errors

        # Check for common problematic types in metadata
        for key, value in metadata.items():
            if not isinstance(key, str):
                errors.append(f"Metadata key must be string: {key}")
            # Check for complex nested structures that might cause serialization issues
            if isinstance(value, (list, dict)) and len(str(value)) > 10000:
                errors.append(f"Metadata value for key '{key}' is too large")

        return errors

    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text input by removing potentially harmful content

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text
        """
        if not text:
            return ""

        # Remove null bytes and other control characters that might be problematic
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        # Remove potentially dangerous patterns (basic protection against injection)
        # This is a basic implementation - in production, use more sophisticated sanitization
        sanitized = sanitized.replace('\0', '')  # Null bytes
        sanitized = sanitized.replace('\x00', '')  # Alternative null byte representation

        return sanitized.strip()