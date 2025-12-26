"""
Qdrant client configuration and service
Handles connection and operations with Qdrant Cloud
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
import uuid
from ..config import Config
from ..utils.logging_config import get_logger
from ..utils.exceptions import VectorStorageError
from ..models.embedding_vector import EmbeddingVector
from ..models.vector_record import VectorRecord


class QdrantService:
    """
    Service class for interacting with Qdrant Cloud
    """
    def __init__(self):
        self.logger = get_logger()
        self.config = Config
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Qdrant client with configuration"""
        try:
            self.client = QdrantClient(
                url=self.config.QDRANT_URL,
                api_key=self.config.QDRANT_API_KEY,
                prefer_grpc=False,  # Using HTTP for better compatibility
                timeout=60  # Increase timeout to 60 seconds for large operations
            )
            self.logger.info(f"Connected to Qdrant at {self.config.QDRANT_URL}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise VectorStorageError(f"Failed to initialize Qdrant client: {str(e)}")

    def create_collection(self, vector_size: int = 1024):
        """
        Create a collection in Qdrant for storing embeddings

        Args:
            vector_size: Size of the embedding vectors (default 1024 for Cohere multilingual model)
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]

            if self.config.QDRANT_COLLECTION_NAME in collection_names:
                self.logger.info(f"Collection {self.config.QDRANT_COLLECTION_NAME} already exists")
                return

            # Create collection with specified vector size
            self.client.create_collection(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            self.logger.info(f"Created collection {self.config.QDRANT_COLLECTION_NAME} with vector size {vector_size}")
        except Exception as e:
            self.logger.error(f"Failed to create collection: {str(e)}")
            raise VectorStorageError(f"Failed to create collection: {str(e)}")

    def store_embeddings(self, embedding_vectors: List[EmbeddingVector], metadata_list: List[Dict[str, Any]] = None):
        """
        Store embedding vectors in Qdrant

        Args:
            embedding_vectors: List of EmbeddingVector objects to store
            metadata_list: Optional list of metadata for each vector
        """
        try:
            if not embedding_vectors:
                self.logger.warning("No embeddings to store")
                return

            # Prepare points for insertion
            points = []
            for i, emb_vec in enumerate(embedding_vectors):
                # Get metadata for this embedding
                metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}

                # Add additional metadata from the embedding vector
                metadata.update({
                    "original_id": emb_vec.id,  # Store original ID in metadata
                    "chunk_id": emb_vec.chunk_id,
                    "model": emb_vec.model,
                    "timestamp": emb_vec.timestamp.isoformat()
                })

                # Generate a valid UUID for the point ID as required by Qdrant
                point_id = str(uuid.uuid4())

                point = models.PointStruct(
                    id=point_id,
                    vector=emb_vec.vector,
                    payload=metadata
                )
                points.append(point)

            # Upload points to Qdrant
            self.client.upsert(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                points=points
            )

            self.logger.info(f"Successfully stored {len(embedding_vectors)} embeddings in Qdrant")
        except Exception as e:
            self.logger.error(f"Failed to store embeddings: {str(e)}")
            raise VectorStorageError(f"Failed to store embeddings: {str(e)}")

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
            search_results = self.client.search(
                collection_name=self.config.QDRANT_COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k
            )

            results = []
            for result in search_results:
                payload = result.payload or {}
                vector_record = VectorRecord(
                    chunk_id=payload.get('chunk_id', ''),
                    content=payload.get('content', ''),
                    source_url=payload.get('source_url', ''),
                    chapter=payload.get('chapter'),
                    section=payload.get('section'),
                    similarity_score=result.score,
                    metadata=payload,
                    timestamp=payload.get('timestamp')
                )
                results.append(vector_record)

            self.logger.info(f"Found {len(results)} similar vectors")
            return results
        except Exception as e:
            self.logger.error(f"Failed to search similar vectors: {str(e)}")
            raise VectorStorageError(f"Failed to search similar vectors: {str(e)}")

    def delete_collection(self):
        """Delete the collection from Qdrant (useful for testing/updates)"""
        try:
            self.client.delete_collection(self.config.QDRANT_COLLECTION_NAME)
            self.logger.info(f"Deleted collection {self.config.QDRANT_COLLECTION_NAME}")
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {str(e)}")
            raise VectorStorageError(f"Failed to delete collection: {str(e)}")

    def get_collection_info(self):
        """Get information about the collection"""
        try:
            collection_info = self.client.get_collection(self.config.QDRANT_COLLECTION_NAME)
            return collection_info
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {str(e)}")
            raise VectorStorageError(f"Failed to get collection info: {str(e)}")