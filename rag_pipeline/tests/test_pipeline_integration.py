"""
Integration tests for the complete RAG pipeline
"""
import unittest
from unittest.mock import Mock, patch
from rag_pipeline.pipelines.rag_pipeline import RAGPipeline
from rag_pipeline.models.book_content import BookContent
from rag_pipeline.models.content_chunk import ContentChunk
from rag_pipeline.models.embedding_vector import EmbeddingVector


class TestRAGPipelineIntegration(unittest.TestCase):
    def setUp(self):
        # Mock all services to avoid external dependencies during testing
        self.mock_extraction_pipeline = Mock()
        self.mock_embedding_generator = Mock()
        self.mock_vector_storage = Mock()
        self.mock_qdrant_schema_manager = Mock()

        with patch('rag_pipeline.pipelines.rag_pipeline.ExtractionPipeline') as mock_ext:
            with patch('rag_pipeline.pipelines.rag_pipeline.EmbeddingGenerator') as mock_emb:
                with patch('rag_pipeline.pipelines.rag_pipeline.VectorStorage') as mock_vs:
                    with patch('rag_pipeline.pipelines.rag_pipeline.QdrantSchemaManager') as mock_qsm:
                        mock_ext.return_value = self.mock_extraction_pipeline
                        mock_emb.return_value = self.mock_embedding_generator
                        mock_vs.return_value = self.mock_vector_storage
                        mock_qsm.return_value = self.mock_qdrant_schema_manager

                        self.rag_pipeline = RAGPipeline()

    def test_run_complete_pipeline_success(self):
        """Test the complete pipeline end-to-end"""
        # Setup mock return values
        content_chunks = [
            ContentChunk(
                id="chunk_1",
                book_content_id="content_1",
                content="Test content for chunking",
                chunk_index=0,
                word_count=5,
                char_count=25
            )
        ]

        embedding_vectors = [
            EmbeddingVector(
                id="emb_1",
                chunk_id="chunk_1",
                vector=[0.1, 0.2, 0.3],
                model="test_model"
            )
        ]

        # Configure mocks
        self.mock_extraction_pipeline.run_extraction_pipeline.return_value = content_chunks
        self.mock_embedding_generator.generate_embeddings.return_value = embedding_vectors
        self.mock_vector_storage.store_embeddings.return_value = True
        self.mock_qdrant_schema_manager.create_book_content_schema.return_value = None

        # Run the pipeline
        results = self.rag_pipeline.run_complete_pipeline()

        # Verify the results
        self.assertEqual(results["status"], "success")
        self.assertEqual(results["content_chunks_count"], 1)
        self.assertEqual(results["embeddings_count"], 1)
        self.assertEqual(results["storage_count"], 1)

        # Verify that each step was called
        self.mock_qdrant_schema_manager.create_book_content_schema.assert_called_once()
        self.mock_extraction_pipeline.run_extraction_pipeline.assert_called_once()
        self.mock_embedding_generator.generate_embeddings.assert_called_once_with(content_chunks)
        self.mock_vector_storage.store_embeddings.assert_called_once()

    def test_run_complete_pipeline_with_validation(self):
        """Test the complete pipeline with validation enabled"""
        content_chunks = [
            ContentChunk(
                id="chunk_1",
                book_content_id="content_1",
                content="Test content for validation",
                chunk_index=0,
                word_count=6,
                char_count=30
            )
        ]

        embedding_vectors = [
            EmbeddingVector(
                id="emb_1",
                chunk_id="chunk_1",
                vector=[0.5, 0.6, 0.7],
                model="test_model"
            )
        ]

        # Configure mocks
        self.mock_extraction_pipeline.run_extraction_pipeline.return_value = content_chunks
        self.mock_embedding_generator.generate_embeddings.return_value = embedding_vectors
        self.mock_vector_storage.store_embeddings.return_value = True
        self.mock_qdrant_schema_manager.create_book_content_schema.return_value = None

        # Mock validation to return successful validation
        self.mock_extraction_pipeline.validate_pipeline_output.return_value = True
        self.mock_embedding_generator.validate_embeddings.return_value = True

        results = self.rag_pipeline.run_complete_pipeline(validate=True)

        self.assertEqual(results["status"], "success")
        self.assertTrue(results["validation"]["valid"])

    def test_pipeline_with_empty_content(self):
        """Test pipeline behavior with no content"""
        # Configure mocks to return empty lists
        self.mock_extraction_pipeline.run_extraction_pipeline.return_value = []
        self.mock_qdrant_schema_manager.create_book_content_schema.return_value = None

        results = self.rag_pipeline.run_complete_pipeline()

        # Should return early with "no_content" status
        self.assertEqual(results["status"], "no_content")
        self.assertEqual(results["content_chunks_count"], 0)

    def test_pipeline_with_embedding_failure(self):
        """Test pipeline behavior when embedding generation fails"""
        content_chunks = [
            ContentChunk(
                id="chunk_1",
                book_content_id="content_1",
                content="Test content",
                chunk_index=0,
                word_count=3,
                char_count=15
            )
        ]

        # Configure mocks
        self.mock_extraction_pipeline.run_extraction_pipeline.return_value = content_chunks
        self.mock_embedding_generator.generate_embeddings.return_value = []  # Simulate failure
        self.mock_qdrant_schema_manager.create_book_content_schema.return_value = None

        results = self.rag_pipeline.run_complete_pipeline()

        # Should return with "embedding_failure" status
        self.assertEqual(results["status"], "embedding_failure")
        self.assertEqual(results["content_chunks_count"], 1)
        self.assertEqual(results["embeddings_count"], 0)


if __name__ == '__main__':
    unittest.main()