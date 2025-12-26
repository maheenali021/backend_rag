"""
Unit tests for vector storage module
"""
import unittest
from unittest.mock import Mock, patch
from rag_pipeline.services.vector_storage import VectorStorage
from rag_pipeline.models.embedding_vector import EmbeddingVector


class TestVectorStorage(unittest.TestCase):
    def setUp(self):
        with patch('rag_pipeline.services.vector_storage.QdrantService'):
            self.vector_storage = VectorStorage()

    def test_store_embeddings_success(self):
        """Test successful storage of embeddings"""
        embedding_vectors = [
            EmbeddingVector(
                id="emb_1",
                chunk_id="chunk_1",
                vector=[0.1, 0.2, 0.3],
                model="test_model"
            ),
            EmbeddingVector(
                id="emb_2",
                chunk_id="chunk_2",
                vector=[0.4, 0.5, 0.6],
                model="test_model"
            )
        ]

        # Mock the Qdrant service
        self.vector_storage.qdrant_service.store_embeddings = Mock(return_value=None)

        result = self.vector_storage.store_embeddings(embedding_vectors)
        self.assertTrue(result)

    def test_store_embeddings_empty_list(self):
        """Test storage with empty embedding list"""
        result = self.vector_storage.store_embeddings([])
        self.assertTrue(result)  # Should return True for empty list

    def test_attach_metadata(self):
        """Test attaching metadata to embedding vectors"""
        embedding_vectors = [
            EmbeddingVector(
                id="emb_1",
                chunk_id="chunk_1",
                vector=[0.1, 0.2],
                model="test_model"
            )
        ]

        source_metadata = [
            {
                "source_url": "http://example.com",
                "chapter": "Chapter 1",
                "section": "Section 1.1"
            }
        ]

        metadata_list = self.vector_storage.attach_metadata(embedding_vectors, source_metadata)

        self.assertEqual(len(metadata_list), 1)
        self.assertEqual(metadata_list[0]["chunk_id"], "chunk_1")
        self.assertEqual(metadata_list[0]["source_url"], "http://example.com")
        self.assertEqual(metadata_list[0]["chapter"], "Chapter 1")

    def test_get_storage_stats(self):
        """Test retrieval of storage statistics"""
        # Mock the Qdrant service to return stats
        mock_collection_info = Mock()
        mock_collection_info.points_count = 100
        self.vector_storage.qdrant_service.get_collection_info = Mock(return_value=mock_collection_info)

        stats = self.vector_storage.get_storage_stats()

        self.assertEqual(stats["vectors_count"], 100)

    def test_validate_storage_reliability(self):
        """Test storage reliability validation"""
        test_embeddings = [
            EmbeddingVector(
                id="emb_1",
                chunk_id="chunk_1",
                vector=[0.1, 0.2],
                model="test_model"
            )
        ]

        # Mock the storage method to return success
        self.vector_storage.store_embeddings = Mock(return_value=True)
        self.vector_storage.get_storage_stats = Mock(return_value={"vectors_count": 1})

        reliability = self.vector_storage.validate_storage_reliability(test_embeddings)
        # Should be 1.0 if 1 out of 1 embeddings was stored
        self.assertEqual(reliability, 1.0)


if __name__ == '__main__':
    unittest.main()