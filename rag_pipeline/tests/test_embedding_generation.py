"""
Unit tests for embedding generation module
"""
import unittest
from unittest.mock import Mock, patch
from rag_pipeline.services.embedding_generator import EmbeddingGenerator
from rag_pipeline.models.content_chunk import ContentChunk
from rag_pipeline.models.embedding_vector import EmbeddingVector


class TestEmbeddingGenerator(unittest.TestCase):
    def setUp(self):
        with patch('rag_pipeline.services.embedding_generator.CohereService'):
            self.embedding_generator = EmbeddingGenerator()

    def test_generate_single_embedding(self):
        """Test generation of a single embedding"""
        content_chunk = ContentChunk(
            id="test_chunk",
            book_content_id="test_content",
            content="This is a test content for embedding",
            chunk_index=0,
            word_count=8,
            char_count=37
        )

        # Mock the Cohere service to return a test embedding
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.embedding_generator.cohere_service.generate_single_embedding = Mock(return_value=test_embedding)

        result = self.embedding_generator.generate_single_embedding(content_chunk)

        self.assertIsInstance(result, EmbeddingVector)
        self.assertEqual(result.chunk_id, "test_chunk")
        self.assertEqual(result.vector, test_embedding)
        self.assertIsNotNone(result.id)

    @patch('rag_pipeline.services.embedding_generator.batch_list')
    def test_generate_embeddings_batch(self, mock_batch_list):
        """Test generation of embeddings for multiple chunks"""
        content_chunks = [
            ContentChunk(
                id="chunk_1",
                book_content_id="test_content",
                content="First chunk content",
                chunk_index=0,
                word_count=3,
                char_count=19
            ),
            ContentChunk(
                id="chunk_2",
                book_content_id="test_content",
                content="Second chunk content",
                chunk_index=1,
                word_count=3,
                char_count=20
            )
        ]

        # Mock the batching function
        mock_batch_list.return_value = [content_chunks]  # Return as one batch

        # Mock the Cohere service to return test embeddings
        test_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        self.embedding_generator.cohere_service.generate_embeddings = Mock(return_value=test_embeddings)
        self.embedding_generator.cohere_service.validate_api_access = Mock(return_value=True)

        results = self.embedding_generator.generate_embeddings(content_chunks)

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, EmbeddingVector)

    def test_validate_embeddings_valid(self):
        """Test validation of valid embeddings"""
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

        is_valid = self.embedding_generator.validate_embeddings(embedding_vectors)
        self.assertTrue(is_valid)

    def test_validate_embeddings_invalid(self):
        """Test validation of invalid embeddings"""
        embedding_vectors = [
            EmbeddingVector(
                id="emb_1",
                chunk_id="chunk_1",
                vector=[],  # Empty vector
                model="test_model"
            ),
            EmbeddingVector(
                id="emb_2",
                chunk_id="chunk_2",
                vector=[0.4, float('nan'), 0.6],  # Contains NaN
                model="test_model"
            )
        ]

        is_valid = self.embedding_generator.validate_embeddings(embedding_vectors)
        self.assertFalse(is_valid)

    def test_get_embedding_stats(self):
        """Test embedding statistics calculation"""
        embedding_vectors = [
            EmbeddingVector(
                id="emb_1",
                chunk_id="chunk_1",
                vector=[0.1, 0.2, 0.3, 0.4],
                model="test_model"
            ),
            EmbeddingVector(
                id="emb_2",
                chunk_id="chunk_2",
                vector=[0.5, 0.6, 0.7, 0.8],
                model="test_model"
            )
        ]

        stats = self.embedding_generator.get_embedding_stats(embedding_vectors)

        self.assertEqual(stats["total_embeddings"], 2)
        self.assertEqual(stats["avg_vector_length"], 4)
        self.assertEqual(stats["model_used"], "test_model")
        self.assertEqual(stats["vector_dimensionality"], 4)


if __name__ == '__main__':
    unittest.main()