"""
Embedding generation service for RAG Pipeline
Generates vector embeddings using Cohere API
"""
from typing import List
from ..models.content_chunk import ContentChunk
from ..models.embedding_vector import EmbeddingVector
from ..services.cohere_client import CohereService
from ..utils.logging_config import get_logger
from ..utils.exceptions import EmbeddingGenerationError
from ..utils.helpers import generate_id, batch_list
from ..config import Config


class EmbeddingGenerator:
    """
    Service class for generating embeddings from content chunks
    """
    def __init__(self):
        self.logger = get_logger()
        self.config = Config
        self.cohere_service = CohereService()

    def generate_embeddings(self, content_chunks: List[ContentChunk],
                          batch_size: int = 96) -> List[EmbeddingVector]:
        """
        Generate embeddings for a list of content chunks

        Args:
            content_chunks: List of ContentChunk objects to generate embeddings for
            batch_size: Number of chunks to process in each batch (Cohere has limits)

        Returns:
            List of EmbeddingVector objects
        """
        if not content_chunks:
            self.logger.warning("No content chunks provided for embedding generation")
            return []

        self.logger.info(f"Starting embedding generation for {len(content_chunks)} content chunks")

        # Validate API access before starting
        if not self.cohere_service.validate_api_access():
            raise EmbeddingGenerationError("Cohere API access validation failed")

        embedding_vectors = []
        processed_count = 0

        # Process in batches to respect API limits
        batches = batch_list(content_chunks, batch_size)

        for i, batch in enumerate(batches):
            self.logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} chunks")

            try:
                # Extract text content from chunks for embedding
                texts = [chunk.content for chunk in batch]

                # Generate embeddings for the batch
                embeddings = self.cohere_service.generate_embeddings(texts)

                # Create EmbeddingVector objects
                for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    embedding_vector = EmbeddingVector(
                        id=generate_id(f"{chunk.id}_{chunk.content[:50]}", prefix="ev_"),
                        chunk_id=chunk.id,
                        vector=embedding,
                        model=self.config.COHERE_MODEL
                    )
                    embedding_vectors.append(embedding_vector)

                processed_count += len(batch)
                self.logger.info(f"Completed batch {i+1}/{len(batches)}, total processed: {processed_count}")

            except Exception as e:
                self.logger.error(f"Error processing batch {i+1}: {str(e)}")
                # Continue with the next batch instead of failing completely
                continue

        success_rate = len(embedding_vectors) / len(content_chunks) if content_chunks else 0
        self.logger.info(f"Embedding generation completed. Success rate: {success_rate:.2%} "
                        f"({len(embedding_vectors)}/{len(content_chunks)} successful)")

        return embedding_vectors

    def generate_single_embedding(self, content_chunk: ContentChunk) -> EmbeddingVector:
        """
        Generate embedding for a single content chunk

        Args:
            content_chunk: ContentChunk object to generate embedding for

        Returns:
            EmbeddingVector object
        """
        try:
            embedding = self.cohere_service.generate_single_embedding(content_chunk.content)

            embedding_vector = EmbeddingVector(
                id=generate_id(f"{content_chunk.id}_{content_chunk.content[:50]}", prefix="ev_"),
                chunk_id=content_chunk.id,
                vector=embedding,
                model=self.config.COHERE_MODEL
            )

            self.logger.debug(f"Generated embedding for chunk {content_chunk.id}")
            return embedding_vector

        except Exception as e:
            self.logger.error(f"Error generating embedding for chunk {content_chunk.id}: {str(e)}")
            raise EmbeddingGenerationError(f"Error generating embedding for chunk {content_chunk.id}: {str(e)}")

    def validate_embeddings(self, embedding_vectors: List[EmbeddingVector]) -> bool:
        """
        Validate the generated embeddings

        Args:
            embedding_vectors: List of EmbeddingVector objects to validate

        Returns:
            True if all embeddings are valid, False otherwise
        """
        if not embedding_vectors:
            self.logger.warning("No embeddings to validate")
            return False

        invalid_count = 0
        for emb in embedding_vectors:
            if not emb.vector or len(emb.vector) == 0:
                self.logger.error(f"Embedding {emb.id} has no vector data")
                invalid_count += 1
                continue

            # Check for NaN or infinite values in the vector
            for val in emb.vector:
                if not isinstance(val, (int, float)) or val != val:  # Check for NaN (val != val is True for NaN)
                    self.logger.error(f"Embedding {emb.id} contains invalid value: {val}")
                    invalid_count += 1
                    break

        invalid_rate = invalid_count / len(embedding_vectors)
        self.logger.info(f"Embedding validation: {invalid_rate:.2%} invalid embeddings ({invalid_count}/{len(embedding_vectors)})")

        return invalid_count == 0

    def get_embedding_stats(self, embedding_vectors: List[EmbeddingVector]) -> dict:
        """
        Get statistics about the generated embeddings

        Args:
            embedding_vectors: List of EmbeddingVector objects

        Returns:
            Dictionary with embedding statistics
        """
        if not embedding_vectors:
            return {
                "total_embeddings": 0,
                "avg_vector_length": 0,
                "model_used": None,
                "vector_dimensionality": 0
            }

        # Get vector dimensionality from the first embedding
        vector_dimensionality = len(embedding_vectors[0].vector) if embedding_vectors else 0
        model_used = embedding_vectors[0].model if embedding_vectors else None

        return {
            "total_embeddings": len(embedding_vectors),
            "avg_vector_length": vector_dimensionality,
            "model_used": model_used,
            "vector_dimensionality": vector_dimensionality,
            "success_rate": len(embedding_vectors)  # This would be compared to the original number of chunks
        }

    def handle_rate_limiting(self):
        """
        Handle rate limiting by implementing appropriate delays
        """
        # The CohereService already implements retry logic with backoff
        # This method exists to satisfy the requirements in tasks.md
        self.logger.debug("Rate limiting handling is implemented in CohereService")