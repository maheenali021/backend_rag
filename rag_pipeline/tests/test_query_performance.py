"""
Performance tests for query functionality
"""
import unittest
import time
from unittest.mock import Mock, patch
from rag_pipeline.services.semantic_search import SemanticSearch
from rag_pipeline.models.vector_record import VectorRecord


class TestQueryPerformance(unittest.TestCase):
    def setUp(self):
        # Mock the services to focus on performance testing logic
        with patch('rag_pipeline.services.semantic_search.CohereService'):
            with patch('rag_pipeline.services.semantic_search.VectorStorage'):
                self.semantic_search = SemanticSearch()

    def test_search_response_time(self):
        """Test that search responses are within performance thresholds"""
        # Mock the search functionality
        mock_results = [
            VectorRecord(
                chunk_id="chunk_1",
                content="Test content result",
                source_url="http://example.com",
                chapter="Chapter 1",
                section="Section 1.1",
                similarity_score=0.8
            )
        ] * 5  # 5 results

        self.semantic_search.vector_storage.search_similar = Mock(return_value=mock_results)
        self.semantic_search.cohere_service.generate_single_embedding = Mock(return_value=[0.1, 0.2, 0.3])

        # Measure search time
        start_time = time.time()
        results = self.semantic_search.search("test query", top_k=5)
        end_time = time.time()

        search_time = end_time - start_time

        # Performance requirement: SC-006 - queries return within 2 seconds
        self.assertLess(search_time, 2.0, f"Search took {search_time:.2f}s, which exceeds 2s limit")

        # Verify we got results
        self.assertEqual(len(results), 5)

    def test_batch_search_performance(self):
        """Test batch search performance"""
        mock_results = [
            VectorRecord(
                chunk_id="chunk_1",
                content="Test content result",
                source_url="http://example.com",
                chapter="Chapter 1",
                similarity_score=0.8
            )
        ]

        self.semantic_search.search = Mock(return_value=mock_results)
        self.semantic_search.cohere_service.generate_single_embedding = Mock(return_value=[0.1, 0.2, 0.3])

        queries = ["query 1", "query 2", "query 3"]

        start_time = time.time()
        results = self.semantic_search.batch_search(queries, top_k=3)
        end_time = time.time()

        search_time = end_time - start_time

        # Should handle multiple queries in reasonable time
        self.assertLess(search_time, 5.0, f"Batch search took {search_time:.2f}s, which exceeds 5s limit")
        self.assertEqual(len(results), 3)  # Should have results for each query

    def test_search_with_context_performance(self):
        """Test search with context performance"""
        mock_results = [
            VectorRecord(
                chunk_id="chunk_1",
                content="This is a longer content with multiple sentences. The second sentence adds more context. Finally, a third sentence completes the thought.",
                source_url="http://example.com",
                chapter="Chapter 1",
                section="Section 1.1",
                similarity_score=0.85
            )
        ]

        self.semantic_search.search = Mock(return_value=mock_results)
        self.semantic_search.cohere_service.generate_single_embedding = Mock(return_value=[0.1, 0.2, 0.3])

        start_time = time.time()
        results = self.semantic_search.search_with_context("test query", top_k=1, context_window=2)
        end_time = time.time()

        search_time = end_time - start_time

        # Should return results with context in reasonable time
        self.assertLess(search_time, 2.0, f"Search with context took {search_time:.2f}s, which exceeds 2s limit")
        self.assertEqual(len(results), 1)
        self.assertIn("context", results[0])

    def test_find_related_content_performance(self):
        """Test related content search performance"""
        mock_results = [
            VectorRecord(
                chunk_id="related_chunk_1",
                content="Related content here",
                source_url="http://example.com",
                chapter="Chapter 2",
                similarity_score=0.75
            )
        ]

        self.semantic_search.vector_storage.search_similar = Mock(return_value=mock_results)
        self.semantic_search.cohere_service.generate_single_embedding = Mock(return_value=[0.4, 0.5, 0.6])

        start_time = time.time()
        results = self.semantic_search.find_related_content("sample content text", top_k=3)
        end_time = time.time()

        search_time = end_time - start_time

        # Should return related content in reasonable time
        self.assertLess(search_time, 2.0, f"Related content search took {search_time:.2f}s, which exceeds 2s limit")
        self.assertEqual(len(results), 1)


if __name__ == '__main__':
    unittest.main()