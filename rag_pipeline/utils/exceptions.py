"""
Custom exception classes for RAG Pipeline
Defines specific exception types for different error scenarios
"""


class RAGPipelineError(Exception):
    """Base exception class for RAG Pipeline errors"""
    pass


class ContentExtractionError(RAGPipelineError):
    """Raised when content extraction fails"""
    pass


class ContentChunkingError(RAGPipelineError):
    """Raised when content chunking fails"""
    pass


class EmbeddingGenerationError(RAGPipelineError):
    """Raised when embedding generation fails"""
    pass


class VectorStorageError(RAGPipelineError):
    """Raised when vector storage operations fail"""
    pass


class ConfigurationError(RAGPipelineError):
    """Raised when configuration is invalid or missing"""
    pass


class APIError(RAGPipelineError):
    """Raised when API calls fail"""
    pass


class ValidationError(RAGPipelineError):
    """Raised when validation fails"""
    pass