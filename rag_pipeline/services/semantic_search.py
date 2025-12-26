"""
Semantic search service for RAG Pipeline
Provides semantic search capabilities using vector embeddings
"""
from typing import List, Dict, Any
from ..models.vector_record import VectorRecord
from ..services.cohere_client import CohereService
from ..services.vector_storage import VectorStorage
from ..utils.logging_config import get_logger
from ..utils.exceptions import RAGPipelineError


class SemanticSearch:
    """
    Service class for performing semantic search using vector embeddings
    """
    def __init__(self):
        self.logger = get_logger()
        self.cohere_service = CohereService()
        self.vector_storage = VectorStorage()

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[VectorRecord]:
        """
        Perform semantic search for a query

        Args:
            query: Search query string
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of VectorRecord objects with similarity scores
        """
        try:
            self.logger.info(f"Performing semantic search for query: '{query[:50]}...'")

            # Generate embedding for the query
            query_embedding = self.cohere_service.generate_single_embedding(query)

            # Search for similar vectors in storage
            results = self.vector_storage.search_similar(query_embedding, top_k)

            # Filter by minimum similarity if specified
            if min_similarity > 0:
                results = [result for result in results if result.similarity_score >= min_similarity]

            self.logger.info(f"Semantic search returned {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"Error performing semantic search: {str(e)}")
            raise RAGPipelineError(f"Error performing semantic search: {str(e)}")

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[VectorRecord]]:
        """
        Perform semantic search for multiple queries

        Args:
            queries: List of search query strings
            top_k: Number of results to return for each query

        Returns:
            List of lists of VectorRecord objects
        """
        results = []
        for query in queries:
            try:
                query_results = self.search(query, top_k)
                results.append(query_results)
            except Exception as e:
                self.logger.error(f"Error searching for query '{query[:30]}...': {str(e)}")
                results.append([])  # Return empty list for failed queries

        return results

    def search_with_context(self, query: str, top_k: int = 5, context_window: int = 2) -> List[Dict[str, Any]]:
        """
        Perform semantic search and return results with additional context

        Args:
            query: Search query string
            top_k: Number of results to return
            context_window: Number of sentences to include as context around matches

        Returns:
            List of dictionaries with search results and context
        """
        try:
            results = self.search(query, top_k)

            enhanced_results = []
            for result in results:
                enhanced_result = {
                    "content": result.content,
                    "source_url": result.source_url,
                    "chapter": result.chapter,
                    "section": result.section,
                    "similarity_score": result.similarity_score,
                    "chunk_id": result.chunk_id,
                    "metadata": result.metadata,
                    "context": self._extract_context(result.content, query, context_window)
                }
                enhanced_results.append(enhanced_result)

            return enhanced_results

        except Exception as e:
            self.logger.error(f"Error performing search with context: {str(e)}")
            raise RAGPipelineError(f"Error performing search with context: {str(e)}")

    def _extract_context(self, content: str, query: str, context_window: int) -> str:
        """
        Extract context around the matching content

        Args:
            content: Full content text
            query: Query string to center context around
            context_window: Number of sentences to include as context

        Returns:
            Context string
        """
        # Simple implementation: split by sentences and find query-relevant sentences
        sentences = content.split('. ')

        # Find sentences that are most relevant to the query
        query_words = set(query.lower().split())
        relevant_sentences = []

        for i, sentence in enumerate(sentences):
            sentence_words = set(sentence.lower().split())
            if query_words.intersection(sentence_words):
                # Include context_window sentences before and after
                start = max(0, i - context_window)
                end = min(len(sentences), i + context_window + 1)
                context = '. '.join(sentences[start:end])
                relevant_sentences.append(context)

        if relevant_sentences:
            return ' ... '.join(relevant_sentences)
        else:
            # If no exact match, return the first few sentences as context
            return '. '.join(sentences[:context_window + 1])

    def find_related_content(self, content: str, top_k: int = 5) -> List[VectorRecord]:
        """
        Find content related to a given text

        Args:
            content: Text to find related content for
            top_k: Number of results to return

        Returns:
            List of VectorRecord objects
        """
        try:
            # Generate embedding for the input content
            content_embedding = self.cohere_service.generate_single_embedding(content)

            # Search for similar vectors in storage
            results = self.vector_storage.search_similar(content_embedding, top_k)

            return results

        except Exception as e:
            self.logger.error(f"Error finding related content: {str(e)}")
            raise RAGPipelineError(f"Error finding related content: {str(e)}")

    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about search performance

        Returns:
            Dictionary with search statistics
        """
        # In a real implementation, this would track search metrics
        # For now, we'll return placeholder statistics
        storage_stats = self.vector_storage.get_storage_stats()

        return {
            "indexed_content_count": storage_stats.get("vectors_count", 0),
            "last_search_time": None,  # Would track actual search times in a full implementation
            "average_similarity": None,  # Would track similarity scores in a full implementation
            "search_success_rate": None,  # Would track success rate in a full implementation
        }

    def validate_search_quality(self, test_queries: List[str], expected_results: List[List[str]] = None) -> Dict[str, float]:
        """
        Validate search quality using test queries

        Args:
            test_queries: List of test queries
            expected_results: List of expected result identifiers (optional)

        Returns:
            Dictionary with quality metrics
        """
        correct_matches = 0
        total_evaluated = 0
        similarity_scores = []

        for i, query in enumerate(test_queries):
            try:
                results = self.search(query, top_k=5)

                if expected_results and i < len(expected_results):
                    # Check if expected results appear in the search results
                    result_chunk_ids = [r.chunk_id for r in results]
                    expected_ids = expected_results[i]

                    for exp_id in expected_ids:
                        if exp_id in result_chunk_ids:
                            correct_matches += 1
                        total_evaluated += 1

                # Collect similarity scores
                for result in results:
                    if result.similarity_score is not None:
                        similarity_scores.append(result.similarity_score)

            except Exception as e:
                self.logger.error(f"Error validating search for query '{query}': {str(e)}")

        # Calculate metrics
        accuracy = correct_matches / total_evaluated if total_evaluated > 0 else 0
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        top_3_accuracy = 0  # Placeholder for top-3 accuracy calculation

        return {
            "accuracy": accuracy,
            "average_similarity": avg_similarity,
            "top_3_accuracy": top_3_accuracy,
            "total_evaluated": total_evaluated,
            "correct_matches": correct_matches
        }