"""
RAG Pipeline module initialization
"""
__version__ = "1.0.0"
__author__ = "AI Robotics Textbook Team"

# Import main classes for easy access
from .pipelines.rag_pipeline import RAGPipeline
from .pipelines.extraction_pipeline import ExtractionPipeline
from .services.content_extractor import ContentExtractor
from .services.content_chunker import ContentChunker
from .services.embedding_generator import EmbeddingGenerator
from .services.vector_storage import VectorStorage
from .services.semantic_search import SemanticSearch

__all__ = [
    "RAGPipeline",
    "ExtractionPipeline",
    "ContentExtractor",
    "ContentChunker",
    "EmbeddingGenerator",
    "VectorStorage",
    "SemanticSearch"
]