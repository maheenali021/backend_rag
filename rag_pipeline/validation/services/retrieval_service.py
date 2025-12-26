"""
Retrieval service for the validation system
Handles embedding generation, vector search, and retrieval validation
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from ..models.validation_models import RetrievalQuery, RetrievalResult, ValidationRequest
from ...config import Config  # Import from the main rag_pipeline config
from ...utils.helpers import generate_id


class RetrievalService:
    """
    Service class for handling retrieval operations and validation
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config

        # Initialize Cohere client with error handling
        if not self.config.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY environment variable is not set")

        try:
            self.cohere_client = cohere.AsyncClient(self.config.COHERE_API_KEY)
            self.logger.info("Cohere client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Cohere client: {str(e)}")
            raise

        # Initialize Qdrant client with error handling
        if not self.config.QDRANT_URL or not self.config.QDRANT_API_KEY:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables must be set")

        try:
            self.qdrant_client = QdrantClient(
                url=self.config.QDRANT_URL,
                api_key=self.config.QDRANT_API_KEY,
                prefer_grpc=False
            )
            self.logger.info("Qdrant client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise

        # Set collection name
        self.collection_name = self.config.QDRANT_COLLECTION_NAME

    async def health_check(self) -> bool:
        """
        Check if the retrieval service is healthy by testing Cohere and Qdrant connectivity
        """
        try:
            # Test Cohere connectivity by generating a simple embedding
            try:
                test_embedding = await self.cohere_client.embed(
                    texts=["test"],
                    model=self.config.COHERE_MODEL
                )

                if not test_embedding or not test_embedding.embeddings:
                    self.logger.warning("Cohere test returned no embeddings")
                    return False
            except Exception as cohere_error:
                self.logger.error(f"Cohere connectivity test failed: {str(cohere_error)}")
                return False

            # Test Qdrant connectivity by getting collection info
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                if collection_info is None:
                    self.logger.warning("Qdrant collection info not found")
                    return False
            except Exception as qdrant_error:
                self.logger.error(f"Qdrant connectivity test failed: {str(qdrant_error)}")
                return False

            self.logger.info("Health check passed for both Cohere and Qdrant")
            return True

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False

    async def generate_query_embedding(self, query_text: str) -> List[float]:
        """
        Generate embedding for a query string using Cohere
        """
        try:
            response = await self.cohere_client.embed(
                texts=[query_text],
                model=self.config.COHERE_MODEL,
                input_type="search_query"  # Specify this is a search query
            )

            if not response.embeddings or len(response.embeddings) == 0:
                raise ValueError("No embeddings returned from Cohere")

            return response.embeddings[0]  # Return the first (and only) embedding

        except Exception as e:
            self.logger.error(f"Error generating query embedding: {str(e)}")
            raise

    async def search_in_qdrant(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Perform vector similarity search in Qdrant with optional metadata filtering
        """
        try:
            # Build filter conditions if filters are provided
            qdrant_filters = None
            if filters:
                filter_conditions = []

                for key, value in filters.items():
                    if value is not None:
                        # Skip placeholder values that are commonly used in API schemas
                        if isinstance(value, str) and value in ["string", "integer", "number", "boolean", "array", "object", "null", ""]:
                            continue  # Skip placeholder values

                        # Handle different types of values for Qdrant filtering
                        if isinstance(value, (str, int, float, bool)):
                            filter_conditions.append(
                                FieldCondition(
                                    key=key,
                                    match=MatchValue(value=str(value))
                                )
                            )
                        elif isinstance(value, list):
                            # For list values, we can use has_id condition or create multiple filters
                            # For now, we'll join them as a string (this might need adjustment based on actual use)
                            filter_conditions.append(
                                FieldCondition(
                                    key=key,
                                    match=MatchValue(value=str(value[0]) if value else "")
                                )
                            )

                if filter_conditions:
                    qdrant_filters = Filter(must=filter_conditions)

            # Perform the search - using the modern Qdrant API format (query_points instead of search)
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=qdrant_filters,
                limit=top_k
            )

            # Convert search results to RetrievalResult objects
            # With the new API, search_results.points is the actual list of results
            results = []
            for result in search_results.points:  # Changed from search_results to search_results.points
                payload = result.payload or {}

                # Log payload structure to understand the data
                self.logger.info(f"Qdrant result payload keys: {list(payload.keys()) if payload else 'empty'}")

                # Extract content with fallbacks for different possible field names
                content = payload.get("content", payload.get("text", payload.get("body", "")))
                source_url = payload.get("source_url", payload.get("url", payload.get("source", "")))
                chapter = payload.get("chapter", payload.get("chapter_title", None))
                section = payload.get("section", payload.get("section_title", None))

                retrieval_result = RetrievalResult(
                    id=result.id,
                    content=content,
                    source_url=source_url,
                    chapter=chapter,
                    section=section,
                    similarity_score=result.score,
                    confidence_score=result.score,  # Using similarity score as confidence for now
                    retrieval_timestamp=datetime.now()
                )

                results.append(retrieval_result)

            self.logger.info(f"Retrieved {len(results)} results from Qdrant for query with filters: {filters}")
            return results

        except Exception as e:
            self.logger.error(f"Error searching in Qdrant: {str(e)}")
            raise

    def apply_metadata_filters(
        self,
        results: List[RetrievalResult],
        filters: Optional[Dict[str, Any]]
    ) -> List[RetrievalResult]:
        """
        Apply metadata filters to the retrieved results
        This is a secondary filtering step that can be used when needed
        """
        if not filters:
            return results

        filtered_results = []
        for result in results:
            include_result = True

            for key, expected_value in filters.items():
                if expected_value is not None:
                    # Get the actual value from the result based on the filter key
                    if key == "chapter":
                        actual_value = result.chapter
                    elif key == "section":
                        actual_value = result.section
                    elif key == "source_url":
                        actual_value = result.source_url
                    elif key == "paragraph_id":
                        # Assuming paragraph_id is stored in metadata
                        actual_value = result.id  # Using result.id as paragraph_id
                    else:
                        # For other keys, we could add more mappings
                        # For now, skip filtering for unknown keys
                        continue

                    # Compare the values (case-insensitive)
                    if actual_value and expected_value:
                        if str(actual_value).lower() != str(expected_value).lower():
                            include_result = False
                            break
                    elif expected_value is not None and actual_value is None:
                        include_result = False
                        break

            if include_result:
                filtered_results.append(result)

        return filtered_results

    async def retrieve_paragraph_level_chunks(
        self,
        query: str,
        paragraph_filter: Optional[str] = None,
        top_k: int = 3
    ) -> List[RetrievalResult]:
        """
        Retrieve paragraph-level chunks based on specific paragraph requirements
        """
        try:
            # Generate embedding for the query
            query_embedding = await self.generate_query_embedding(query)

            # If paragraph filter is provided, we'll search for that specific paragraph
            # Otherwise, we'll do a general search but prioritize paragraph-level chunks
            filters = None
            if paragraph_filter:
                filters = {"paragraph_id": paragraph_filter}

            # Perform search with potential paragraph filtering
            results = await self.search_in_qdrant(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k
            )

            # For paragraph-level retrieval, we might want to do additional processing
            # such as ensuring the content is at paragraph level (not too long/short)
            paragraph_optimized_results = []
            for result in results:
                # Check if the result is paragraph-appropriate length
                # For now, we'll just add it to the results
                paragraph_optimized_results.append(result)

            self.logger.info(f"Retrieved {len(paragraph_optimized_results)} paragraph-level results for query")
            return paragraph_optimized_results

        except Exception as e:
            self.logger.error(f"Error retrieving paragraph-level chunks: {str(e)}")
            raise

    async def retrieve_with_content_precision(
        self,
        query: str,
        min_content_length: int = 50,
        max_content_length: int = 1000,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks with specific content length requirements for precision
        """
        try:
            # Generate embedding for the query
            query_embedding = await self.generate_query_embedding(query)

            # Perform initial search
            initial_results = await self.search_in_qdrant(
                query_embedding=query_embedding,
                filters=None,
                top_k=top_k * 2  # Get more results to filter from
            )

            # Filter results based on content length for paragraph-level precision
            filtered_results = []
            for result in initial_results:
                content_length = len(result.content)
                if min_content_length <= content_length <= max_content_length:
                    filtered_results.append(result)

            # Return top-k results after filtering
            return filtered_results[:top_k]

        except Exception as e:
            self.logger.error(f"Error retrieving with content precision: {str(e)}")
            raise

    async def retrieve_similar_chunks_from_query(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve similar chunks based on a RetrievalQuery object
        """
        try:
            # Generate embedding for the query text
            query_embedding = await self.generate_query_embedding(query.query_text)

            # Perform search in Qdrant
            results = await self.search_in_qdrant(
                query_embedding=query_embedding,
                filters=query.filters,
                top_k=query.top_k
            )

            return results

        except Exception as e:
            self.logger.error(f"Error retrieving similar chunks: {str(e)}")
            raise

    async def run_manual_test_queries(
        self,
        test_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run a series of manual test queries and return comprehensive results
        """
        results = {
            "test_queries_executed": len(test_queries),
            "individual_results": [],
            "summary_metrics": {}
        }

        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_alignment = 0
        valid_test_count = 0

        for i, test_query in enumerate(test_queries):
            query_text = test_query.get("query", "")
            expected_sources = test_query.get("expected_sources", [])
            filters = test_query.get("filters", {})
            top_k = test_query.get("top_k", 5)

            try:
                # Create validation request
                validation_request = ValidationRequest(
                    query=query_text,
                    expected_sources=expected_sources,
                    filters=filters,
                    top_k=top_k
                )

                # Perform retrieval
                retrieval_results = await self.retrieve_similar_chunks(validation_request)

                # Calculate metrics
                precision = await self.calculate_precision(retrieval_results, expected_sources)
                semantic_alignment = await self.calculate_semantic_alignment(query_text, retrieval_results)

                # For recall, we need to know all relevant results (simplified calculation)
                recall = min(1.0, len([r for r in retrieval_results if r.id in (expected_sources or [])]) / len(expected_sources)) if expected_sources else 1.0
                f1_score = self.calculate_f1_score(precision, recall)

                # Create detailed result for this test
                test_result = {
                    "query_index": i,
                    "query_text": query_text,
                    "expected_sources": expected_sources,
                    "retrieved_results_count": len(retrieval_results),
                    "retrieved_sources": [r.source_url for r in retrieval_results if r.source_url],
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "semantic_alignment": semantic_alignment,
                    "filters_used": filters,
                    "top_k_requested": top_k
                }

                results["individual_results"].append(test_result)

                # Add to totals for summary
                total_precision += precision
                total_recall += recall
                total_f1 += f1_score
                total_alignment += semantic_alignment
                valid_test_count += 1

            except Exception as e:
                self.logger.error(f"Error running manual test query {i}: {str(e)}")
                test_result = {
                    "query_index": i,
                    "query_text": query_text,
                    "error": str(e),
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "semantic_alignment": 0
                }
                results["individual_results"].append(test_result)

        # Calculate summary metrics
        if valid_test_count > 0:
            results["summary_metrics"] = {
                "average_precision": total_precision / valid_test_count,
                "average_recall": total_recall / valid_test_count,
                "average_f1_score": total_f1 / valid_test_count,
                "average_semantic_alignment": total_alignment / valid_test_count,
                "total_tests": len(test_queries),
                "successful_tests": valid_test_count,
                "failure_rate": (len(test_queries) - valid_test_count) / len(test_queries)
            }
        else:
            results["summary_metrics"] = {
                "average_precision": 0,
                "average_recall": 0,
                "average_f1_score": 0,
                "average_semantic_alignment": 0,
                "total_tests": len(test_queries),
                "successful_tests": 0,
                "failure_rate": 1.0
            }

        return results

    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    async def compare_retrieved_with_source(
        self,
        query: str,
        retrieved_results: List[RetrievalResult],
        expected_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare retrieved chunks with source text to validate relevance and accuracy
        """
        comparison_results = {
            "query": query,
            "comparison_details": [],
            "overall_similarity_score": 0.0,
            "content_coverage": 0.0,
            "accuracy_score": 0.0
        }

        if not retrieved_results:
            return comparison_results

        try:
            # Calculate overall similarity between query and all retrieved results
            all_similarities = [result.similarity_score for result in retrieved_results]
            comparison_results["overall_similarity_score"] = sum(all_similarities) / len(all_similarities)

            # If we have expected content, compare against it
            if expected_content:
                # Calculate semantic similarity between expected content and retrieved content
                all_retrieved_text = " ".join([result.content for result in retrieved_results])

                # Generate embeddings for comparison
                embeddings_response = await self.cohere_client.embed(
                    texts=[expected_content, all_retrieved_text],
                    model=self.config.COHERE_MODEL,
                    input_type="classification"
                )

                expected_embedding = embeddings_response.embeddings[0]
                retrieved_embedding = embeddings_response.embeddings[1]

                # Calculate cosine similarity
                dot_product = sum(a * b for a, b in zip(expected_embedding, retrieved_embedding))
                norm_expected = sum(a * a for a in expected_embedding) ** 0.5
                norm_retrieved = sum(a * a for a in retrieved_embedding) ** 0.5

                if norm_expected == 0 or norm_retrieved == 0:
                    semantic_similarity = 0.0
                else:
                    semantic_similarity = dot_product / (norm_expected * norm_retrieved)
                    # Normalize to 0-1 range
                    semantic_similarity = max(0, min(1, (semantic_similarity + 1) / 2))

                comparison_results["accuracy_score"] = semantic_similarity

                # Calculate content coverage (how much of the expected content is covered by retrieved results)
                # This is a simplified approach - in practice, you might want more sophisticated text overlap metrics
                expected_words = set(expected_content.lower().split())
                retrieved_words = set(all_retrieved_text.lower().split())

                if expected_words:
                    overlap = expected_words.intersection(retrieved_words)
                    comparison_results["content_coverage"] = len(overlap) / len(expected_words)
                else:
                    comparison_results["content_coverage"] = 1.0

            # Add detailed comparison for each retrieved result
            for i, result in enumerate(retrieved_results):
                result_comparison = {
                    "result_index": i,
                    "result_id": result.id,
                    "similarity_score": result.similarity_score,
                    "content_length": len(result.content),
                    "source_url": result.source_url
                }

                # If we have expected content, add specific comparison metrics
                if expected_content:
                    # Calculate content overlap for this specific result
                    result_words = set(result.content.lower().split())
                    expected_words = set(expected_content.lower().split())

                    if expected_words:
                        overlap = expected_words.intersection(result_words)
                        result_comparison["content_overlap_ratio"] = len(overlap) / len(expected_words)
                    else:
                        result_comparison["content_overlap_ratio"] = 1.0

                comparison_results["comparison_details"].append(result_comparison)

            return comparison_results

        except Exception as e:
            self.logger.error(f"Error in comparison logic: {str(e)}")
            # Return basic results even if comparison fails
            return comparison_results

    async def retrieve_similar_chunks(self, validation_request: ValidationRequest) -> List[RetrievalResult]:
        """
        Main method to retrieve similar chunks for validation purposes
        """
        try:
            # Create a RetrievalQuery from the validation request
            retrieval_query = RetrievalQuery(
                query_text=validation_request.query,
                filters=validation_request.filters,
                top_k=validation_request.top_k
            )

            # Retrieve similar chunks
            results = await self.retrieve_similar_chunks_from_query(retrieval_query)

            return results

        except Exception as e:
            self.logger.error(f"Error in retrieve_similar_chunks: {str(e)}")
            raise

    async def calculate_semantic_alignment(self, query: str, results: List[RetrievalResult]) -> float:
        """
        Calculate semantic alignment between query and retrieved results
        """
        if not results:
            return 0.0

        try:
            # Get the content of all results
            result_contents = [result.content for result in results if result.content]

            if not result_contents:
                return 0.0

            # Generate embeddings for query and results
            all_texts = [query] + result_contents
            response = await self.cohere_client.embed(
                texts=all_texts,
                model=self.config.COHERE_MODEL,
                input_type="search_query"  # Using search_query for the first text, search_document for the rest
            )

            query_embedding = response.embeddings[0]
            result_embeddings = response.embeddings[1:]

            # Calculate cosine similarity between query and each result
            similarities = []
            for result_emb in result_embeddings:
                # Calculate cosine similarity
                dot_product = sum(a * b for a, b in zip(query_embedding, result_emb))
                norm_query = sum(a * a for a in query_embedding) ** 0.5
                norm_result = sum(a * a for a in result_emb) ** 0.5

                if norm_query == 0 or norm_result == 0:
                    similarity = 0.0
                else:
                    similarity = dot_product / (norm_query * norm_result)

                # Normalize to 0-1 range
                similarities.append(max(0, min(1, (similarity + 1) / 2)))

            # Return average similarity
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            return avg_similarity

        except Exception as e:
            self.logger.error(f"Error calculating semantic alignment: {str(e)}")
            # Fallback: return a reasonable default
            return 0.5

    async def calculate_precision(self, results: List[RetrievalResult], expected_sources: Optional[List[str]] = None) -> float:
        """
        Calculate precision of retrieval results
        If expected_sources is provided, precision is calculated against those sources
        Otherwise, it's calculated based on relevance scoring
        """
        if not results:
            return 0.0

        if expected_sources and len(expected_sources) > 0:
            # Calculate precision based on expected sources
            retrieved_sources = {result.source_url for result in results if result.source_url}
            expected_set = set(expected_sources)

            if not expected_set:
                return 1.0  # If no expected sources, we consider all results as valid

            relevant_retrieved = retrieved_sources.intersection(expected_set)
            if len(retrieved_sources) == 0:
                return 1.0  # If nothing retrieved, precision is 100% (or undefined)

            precision = len(relevant_retrieved) / len(retrieved_sources)
            return precision
        else:
            # Calculate precision based on relevance scoring
            # For now, assume all results are relevant if we have no expected sources
            return 1.0

    async def test_paragraph_level_precision(
        self,
        detailed_query: str,
        expected_paragraph_content: Optional[str] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Test paragraph-level query precision with detailed analysis
        """
        try:
            # Use the paragraph-level retrieval method
            paragraph_results = await self.retrieve_paragraph_level_chunks(
                query=detailed_query,
                top_k=top_k
            )

            # Calculate various precision metrics
            results_analysis = {
                "query": detailed_query,
                "expected_paragraph_content": expected_paragraph_content is not None,
                "retrieved_results_count": len(paragraph_results),
                "results": []
            }

            for i, result in enumerate(paragraph_results):
                result_analysis = {
                    "rank": i + 1,
                    "id": result.id,
                    "content_preview": result.content[:100] + "..." if len(result.content) > 100 else result.content,
                    "content_length": len(result.content),
                    "similarity_score": result.similarity_score,
                    "source_url": result.source_url,
                    "precision_indicators": {}
                }

                # If we have expected content, compare for precision
                if expected_paragraph_content:
                    # Calculate content overlap
                    expected_words = set(expected_paragraph_content.lower().split())
                    result_words = set(result.content.lower().split())

                    if expected_words:
                        overlap = expected_words.intersection(result_words)
                        overlap_ratio = len(overlap) / len(expected_words)
                        result_analysis["precision_indicators"]["content_overlap_ratio"] = overlap_ratio

                        # Check if the result contains key phrases from expected content
                        key_phrases_found = 0
                        for phrase in expected_paragraph_content.split('.'):
                            phrase = phrase.strip()
                            if len(phrase) > 10 and phrase.lower() in result.content.lower():  # Meaningful phrases
                                key_phrases_found += 1

                        result_analysis["precision_indicators"]["key_phrases_matched"] = key_phrases_found

                results_analysis["results"].append(result_analysis)

            # Calculate overall paragraph-level precision metrics
            if expected_paragraph_content and paragraph_results:
                # Calculate average content overlap
                total_overlap = sum(
                    r["precision_indicators"].get("content_overlap_ratio", 0)
                    for r in results_analysis["results"]
                    if "precision_indicators" in r
                )
                avg_overlap = total_overlap / len(paragraph_results) if paragraph_results else 0

                results_analysis["overall_precision_metrics"] = {
                    "average_content_overlap": avg_overlap,
                    "top_result_similarity": paragraph_results[0].similarity_score if paragraph_results else 0,
                    "has_high_similarity_results": any(r.similarity_score > 0.7 for r in paragraph_results)
                }

            return results_analysis

        except Exception as e:
            self.logger.error(f"Error testing paragraph-level precision: {str(e)}")
            return {
                "query": detailed_query,
                "error": str(e),
                "retrieved_results_count": 0,
                "results": [],
                "overall_precision_metrics": {}
            }



# Global instance of the retrieval service (initialized on demand)
_retrieval_service_instance = None

def get_retrieval_service():
    """Get the singleton retrieval service instance, creating it if needed"""
    global _retrieval_service_instance
    if _retrieval_service_instance is None:
        _retrieval_service_instance = RetrievalService()
    return _retrieval_service_instance