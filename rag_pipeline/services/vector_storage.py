"""
Vector storage service for RAG Pipeline
Handles storage and retrieval of embedding vectors in Qdrant
"""
from typing import List, Dict, Any, Optional
from ..models.embedding_vector import EmbeddingVector
from ..models.vector_record import VectorRecord
from ..services.qdrant_client import QdrantService
from ..utils.logging_config import get_logger
from ..utils.exceptions import VectorStorageError
from ..utils.helpers import batch_list


class VectorStorage:
    """
    Service class for storing and retrieving vector embeddings in Qdrant
    """
    def __init__(self):
        self.logger = get_logger()
        self.qdrant_service = QdrantService()

    def store_embeddings(self, embedding_vectors: List[EmbeddingVector],
                        metadata_list: List[Dict[str, Any]] = None,
                        batch_size: int = 50) -> bool:
        """
        Store embedding vectors in Qdrant

        Args:
            embedding_vectors: List of EmbeddingVector objects to store
            metadata_list: Optional list of metadata for each vector
            batch_size: Number of vectors to store in each batch

        Returns:
            True if storage was successful, False otherwise
        """
        if not embedding_vectors:
            self.logger.warning("No embeddings to store")
            return True

        self.logger.info(f"Starting vector storage for {len(embedding_vectors)} embeddings")

        try:
            # Process in batches to optimize performance
            embedding_batches = batch_list(embedding_vectors, batch_size)

            if metadata_list:
                metadata_batches = batch_list(metadata_list, batch_size)
            else:
                metadata_batches = [None] * len(embedding_batches)

            total_stored = 0
            for i, (emb_batch, meta_batch) in enumerate(zip(embedding_batches, metadata_batches)):
                self.logger.info(f"Storing batch {i+1}/{len(embedding_batches)} with {len(emb_batch)} vectors")

                # Store the batch
                self.qdrant_service.store_embeddings(emb_batch, meta_batch)
                total_stored += len(emb_batch)

            self.logger.info(f"Successfully stored {total_stored} embedding vectors in Qdrant")
            return True

        except Exception as e:
            self.logger.error(f"Error storing embeddings in Qdrant: {str(e)}")
            raise VectorStorageError(f"Error storing embeddings in Qdrant: {str(e)}")

    def attach_metadata(self, embedding_vectors: List[EmbeddingVector],
                      source_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Attach metadata to embedding vectors before storage

        Args:
            embedding_vectors: List of EmbeddingVector objects
            source_metadata: List of source metadata for each vector

        Returns:
            List of metadata dictionaries ready for Qdrant storage
        """
        if len(embedding_vectors) != len(source_metadata):
            raise VectorStorageError(f"Mismatch between embedding vectors ({len(embedding_vectors)}) "
                                   f"and metadata ({len(source_metadata)})")

        metadata_list = []
        for emb, src_meta in zip(embedding_vectors, source_metadata):
            # Combine embedding-specific metadata with source metadata
            combined_meta = {
                "chunk_id": emb.chunk_id,
                "model": emb.model,
                "timestamp": emb.timestamp.isoformat(),
            }

            # Add source metadata if provided
            if src_meta:
                combined_meta.update(src_meta)

            metadata_list.append(combined_meta)

        return metadata_list

    def store_content_chunks_with_metadata(self, embedding_vectors: List[EmbeddingVector],
                                         content_chunks: List[Dict[str, Any]]) -> bool:
        """
        Store embeddings with content chunk metadata

        Args:
            embedding_vectors: List of EmbeddingVector objects to store
            content_chunks: List of content chunk metadata

        Returns:
            True if storage was successful, False otherwise
        """
        if len(embedding_vectors) != len(content_chunks):
            raise VectorStorageError(f"Mismatch between embedding vectors ({len(embedding_vectors)}) "
                                   f"and content chunks ({len(content_chunks)})")

        # Prepare metadata for each embedding
        metadata_list = []
        for emb, chunk in zip(embedding_vectors, content_chunks):
            # Extract relevant information from content chunk
            meta = {
                "chunk_id": emb.chunk_id,
                "content": chunk.get('content', '')[:500],  # Store first 500 chars as context
                "source_url": chunk.get('source_url', ''),
                "chapter": chunk.get('chapter', ''),
                "section": chunk.get('section', ''),
                "book_title": chunk.get('book_title', ''),
                "word_count": chunk.get('word_count', 0),
                "char_count": chunk.get('char_count', 0),
                "model": emb.model,
                "timestamp": emb.timestamp.isoformat()
            }
            metadata_list.append(meta)

        # Store with metadata
        return self.store_embeddings(embedding_vectors, metadata_list)

    def search_similar(self, query_vector: List[float], top_k: int = 10) -> List[VectorRecord]:
        """
        Search for similar vectors in Qdrant

        Args:
            query_vector: Vector to search for similarity
            top_k: Number of similar vectors to return

        Returns:
            List of VectorRecord objects with similarity scores
        """
        try:
            results = self.qdrant_service.search_similar(query_vector, top_k)
            self.logger.info(f"Found {len(results)} similar vectors")
            return results
        except Exception as e:
            self.logger.error(f"Error searching for similar vectors: {str(e)}")
            raise VectorStorageError(f"Error searching for similar vectors: {str(e)}")

    def search_by_content(self, content: str, top_k: int = 10) -> List[VectorRecord]:
        """
        Search for similar content by first generating an embedding for the query content

        Args:
            content: Content to search for similar matches
            top_k: Number of similar vectors to return

        Returns:
            List of VectorRecord objects with similarity scores
        """
        # Note: This would require access to the embedding service to generate
        # a query embedding from the content string
        # For now, this is a placeholder that would need embedding service integration
        raise NotImplementedError(
            "Direct content search requires embedding service integration. "
            "Use search_similar with a pre-generated embedding instead."
        )

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector storage

        Returns:
            Dictionary with storage statistics
        """
        try:
            collection_info = self.qdrant_service.get_collection_info()
            return {
                "vectors_count": collection_info.points_count,
                "collection_name": self.qdrant_service.config.QDRANT_COLLECTION_NAME,
                "config": collection_info.config.dict() if hasattr(collection_info.config, 'dict') else {}
            }
        except Exception as e:
            self.logger.error(f"Error getting storage stats: {str(e)}")
            return {"error": str(e)}

    def validate_storage_reliability(self, test_embeddings: List[EmbeddingVector] = None) -> float:
        """
        Validate storage reliability by performing test operations

        Args:
            test_embeddings: Optional list of test embeddings to use

        Returns:
            Reliability score (0.0 to 1.0)
        """
        # This would involve storing test vectors and verifying they can be retrieved
        # For now, we'll return a placeholder implementation
        try:
            if test_embeddings:
                # Store test embeddings
                success = self.store_embeddings(test_embeddings)
                if not success:
                    return 0.0

                # Verify they can be retrieved (simplified check)
                stats = self.get_storage_stats()
                stored_count = stats.get("vectors_count", 0)
                expected_count = len(test_embeddings)

                return min(1.0, stored_count / expected_count) if expected_count > 0 else 1.0
            else:
                # If no test embeddings provided, check if we can access the collection
                stats = self.get_storage_stats()
                return 1.0 if "error" not in stats else 0.0

        except Exception as e:
            self.logger.error(f"Storage reliability validation failed: {str(e)}")
            return 0.0

    def delete_vectors(self, vector_ids: List[str]) -> bool:
        """
        Delete specific vectors from storage

        Args:
            vector_ids: List of vector IDs to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Note: This would require Qdrant client implementation for deletion
            # which is not currently in the QdrantService
            raise NotImplementedError("Vector deletion not yet implemented in QdrantService")
        except Exception as e:
            self.logger.error(f"Error deleting vectors: {str(e)}")
            return False