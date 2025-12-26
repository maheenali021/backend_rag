"""
Main RAG Pipeline Orchestration
Orchestrates the complete RAG pipeline from content extraction to vector storage
"""
from typing import List, Dict, Any, Optional
from ..models.book_content import BookContent
from ..models.content_chunk import ContentChunk
from ..models.embedding_vector import EmbeddingVector
from ..models.vector_record import VectorRecord
from ..pipelines.extraction_pipeline import ExtractionPipeline
from ..services.embedding_generator import EmbeddingGenerator
from ..services.vector_storage import VectorStorage
from ..services.qdrant_schema import QdrantSchemaManager
from ..utils.logging_config import get_logger
from ..utils.exceptions import RAGPipelineError
from ..utils.error_handler import error_handler, handle_pipeline_error


class RAGPipeline:
    """
    Main orchestration class for the complete RAG pipeline
    """
    def __init__(self):
        self.logger = get_logger()
        self.extraction_pipeline = ExtractionPipeline()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_storage = VectorStorage()
        self.qdrant_schema_manager = QdrantSchemaManager()

    @handle_pipeline_error("RAGPipeline.run_complete_pipeline")
    def run_complete_pipeline(self, website_url: str = None, validate: bool = True) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline from content extraction to vector storage

        Args:
            website_url: URL of the website to process (optional, defaults to config)
            validate: Whether to perform validation steps

        Returns:
            Dictionary with pipeline results and metrics
        """
        self.logger.info("Starting complete RAG pipeline")

        # Recreate the Qdrant collection schema to ensure clean state with content field
        try:
            # First try to recreate the collection (this will delete and create fresh)
            self.qdrant_schema_manager.recreate_collection(vector_size=1024, confirm=True)
        except Exception as e:
            # If recreate fails, try to create normally
            self.logger.warning(f"Collection recreation failed, attempting normal creation: {str(e)}")
            self.qdrant_schema_manager.create_book_content_schema(1024)

        # Step 1: Extract and chunk content
        self.logger.info("Step 1: Extracting and chunking content")
        content_chunks = self.extraction_pipeline.run_extraction_pipeline(website_url)

        if not content_chunks:
            self.logger.warning("No content chunks generated, pipeline stopped early")
            return {
                "status": "no_content",
                "content_chunks_count": 0,
                "embeddings_count": 0,
                "storage_count": 0
            }

        # Step 2: Generate embeddings
        self.logger.info("Step 2: Generating embeddings")
        embedding_vectors = self.embedding_generator.generate_embeddings(content_chunks)

        if not embedding_vectors:
            self.logger.error("No embeddings generated, pipeline failed")
            return {
                "status": "embedding_failure",
                "content_chunks_count": len(content_chunks),
                "embeddings_count": 0,
                "storage_count": 0
            }

        # Step 3: Store embeddings in vector database
        self.logger.info("Step 3: Storing embeddings in vector database")
        success = self._store_embeddings_with_metadata(embedding_vectors, content_chunks)

        if not success:
            self.logger.error("Failed to store embeddings, pipeline failed")
            return {
                "status": "storage_failure",
                "content_chunks_count": len(content_chunks),
                "embeddings_count": len(embedding_vectors),
                "storage_count": 0
            }

        # Validation step
        if validate:
            self.logger.info("Performing validation")
            validation_results = self._validate_pipeline_results(content_chunks, embedding_vectors)
        else:
            validation_results = {"valid": True, "details": "Validation skipped"}

        # Collect metrics
        metrics = self._collect_pipeline_metrics(content_chunks, embedding_vectors)

        results = {
            "status": "success" if validation_results["valid"] else "partial_success",
            "content_chunks_count": len(content_chunks),
            "embeddings_count": len(embedding_vectors),
            "storage_count": len(embedding_vectors),  # Assuming all embeddings were stored
            "validation": validation_results,
            "metrics": metrics
        }

        self.logger.info(f"RAG pipeline completed with status: {results['status']}")
        return results

    def _store_embeddings_with_metadata(self, embedding_vectors: List[EmbeddingVector],
                                      content_chunks: List[ContentChunk]) -> bool:
        """
        Store embeddings with associated metadata from content chunks

        Args:
            embedding_vectors: List of embedding vectors to store
            content_chunks: List of corresponding content chunks

        Returns:
            True if storage was successful, False otherwise
        """
        try:
            # Create metadata for each embedding based on the content chunk
            metadata_list = []
            chunk_map = {chunk.id: chunk for chunk in content_chunks}

            for emb in embedding_vectors:
                chunk = chunk_map.get(emb.chunk_id)
                if chunk:
                    # Get the original book content to extract additional metadata
                    # For this implementation, we'll use placeholder metadata
                    # In a real implementation, we'd have access to the original BookContent
                    meta = {
                        "chunk_id": emb.chunk_id,
                        "content": chunk.content[:500],  # Store first 500 chars as context
                        "source_url": getattr(chunk, 'source_url', ''),
                        "chapter": getattr(chunk, 'chapter', ''),
                        "section": getattr(chunk, 'section', ''),
                        "book_content_id": chunk.book_content_id,
                        "chunk_index": chunk.chunk_index,
                        "word_count": chunk.word_count,
                        "char_count": chunk.char_count,
                        "semantic_boundary": chunk.semantic_boundary,
                        "model": emb.model,
                        "timestamp": emb.timestamp.isoformat()
                    }
                else:
                    # Fallback metadata if chunk not found
                    meta = {
                        "chunk_id": emb.chunk_id,
                        "model": emb.model,
                        "timestamp": emb.timestamp.isoformat()
                    }

                metadata_list.append(meta)

            # Store embeddings with metadata
            return self.vector_storage.store_embeddings(embedding_vectors, metadata_list)

        except Exception as e:
            self.logger.error(f"Error storing embeddings with metadata: {str(e)}")
            return False

    @handle_pipeline_error("RAGPipeline.run_incremental_pipeline")
    def run_incremental_pipeline(self, website_url: str = None) -> Dict[str, Any]:
        """
        Run the RAG pipeline for new or updated content only

        Args:
            website_url: URL of the website to process

        Returns:
            Dictionary with pipeline results and metrics
        """
        self.logger.info("Starting incremental RAG pipeline")

        # For now, this runs a full pipeline
        # In a more sophisticated implementation, this would:
        # 1. Check what content has changed since the last run
        # 2. Only process the changed content
        # 3. Update only the affected embeddings in the vector store

        return self.run_complete_pipeline(website_url, validate=True)

    def _validate_pipeline_results(self, content_chunks: List[ContentChunk],
                                 embedding_vectors: List[EmbeddingVector]) -> Dict[str, Any]:
        """
        Validate the results of the pipeline

        Args:
            content_chunks: List of content chunks processed
            embedding_vectors: List of embedding vectors generated

        Returns:
            Dictionary with validation results
        """
        try:
            # Validate content chunks
            chunk_validation = self.extraction_pipeline.validate_pipeline_output(
                [],  # We don't have BookContent objects here, so pass empty list
                content_chunks
            )

            # Validate embeddings
            embedding_validation = self.embedding_generator.validate_embeddings(embedding_vectors)

            # Check success rates
            expected_chunks = len(content_chunks)
            generated_embeddings = len(embedding_vectors)
            success_rate = generated_embeddings / expected_chunks if expected_chunks > 0 else 0

            # Target: 95% embedding generation success rate
            success_rate_target_met = success_rate >= 0.95

            # Storage validation (check if vectors are retrievable)
            storage_stats = self.vector_storage.get_storage_stats()
            stored_count = storage_stats.get("vectors_count", 0)
            storage_reliability = stored_count >= len(embedding_vectors) * 0.99  # Target: 99% reliability

            validation_results = {
                "valid": (chunk_validation and
                         embedding_validation and
                         success_rate_target_met and
                         storage_reliability),
                "details": {
                    "content_validation": chunk_validation,
                    "embedding_validation": embedding_validation,
                    "success_rate": success_rate,
                    "success_rate_target_met": success_rate_target_met,
                    "storage_reliability": storage_reliability,
                    "stored_count": stored_count,
                    "expected_count": len(embedding_vectors)
                }
            }

            return validation_results

        except Exception as e:
            self.logger.error(f"Error during pipeline validation: {str(e)}")
            return {"valid": False, "details": f"Validation error: {str(e)}"}

    def _collect_pipeline_metrics(self, content_chunks: List[ContentChunk],
                                embedding_vectors: List[EmbeddingVector]) -> Dict[str, Any]:
        """
        Collect metrics about the pipeline execution

        Args:
            content_chunks: List of content chunks processed
            embedding_vectors: List of embedding vectors generated

        Returns:
            Dictionary with pipeline metrics
        """
        # Content metrics
        total_content_chars = sum(len(chunk.content) for chunk in content_chunks)
        avg_chunk_size = total_content_chars / len(content_chunks) if content_chunks else 0

        # Embedding metrics
        if embedding_vectors:
            vector_dimensionality = len(embedding_vectors[0].vector)
        else:
            vector_dimensionality = 0

        # Storage metrics
        storage_stats = self.vector_storage.get_storage_stats()

        return {
            "content": {
                "total_chunks": len(content_chunks),
                "total_characters": total_content_chars,
                "average_chunk_size": avg_chunk_size,
                "total_words": sum(chunk.word_count for chunk in content_chunks)
            },
            "embeddings": {
                "total_embeddings": len(embedding_vectors),
                "vector_dimensionality": vector_dimensionality,
                "generation_success_rate": len(embedding_vectors) / len(content_chunks) if content_chunks else 0
            },
            "storage": storage_stats,
            "performance": {
                # These would be populated with actual timing data in a full implementation
                "total_pipeline_time": None,
                "extraction_time": None,
                "embedding_time": None,
                "storage_time": None
            }
        }

    @handle_pipeline_error("RAGPipeline.query_pipeline")
    def query(self, search_query: str, top_k: int = 5) -> List[VectorRecord]:
        """
        Query the RAG pipeline for relevant content

        Args:
            search_query: Query string
            top_k: Number of results to return

        Returns:
            List of VectorRecord objects with relevant content
        """
        from ..services.semantic_search import SemanticSearch
        semantic_search = SemanticSearch()
        return semantic_search.search(search_query, top_k)

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get the current status of the RAG pipeline

        Returns:
            Dictionary with pipeline status information
        """
        storage_stats = self.vector_storage.get_storage_stats()

        return {
            "pipeline_ready": True,
            "vector_storage": storage_stats,
            "model_info": self.embedding_generator.cohere_service.get_model_info(),
            "content_count": storage_stats.get("vectors_count", 0)
        }