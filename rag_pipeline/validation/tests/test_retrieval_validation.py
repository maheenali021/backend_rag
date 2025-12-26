"""
Unit tests for the retrieval validation system
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from ..models.validation_models import ValidationRequest, ValidationResponse, RetrievalQuery, RetrievalResult
from ..services.retrieval_service import RetrievalService


@pytest.fixture
def mock_retrieval_service():
    """Create a mock retrieval service for testing"""
    service = RetrievalService()

    # Mock the Cohere client
    service.cohere_client = AsyncMock()
    service.cohere_client.embed = AsyncMock(return_value=MagicMock(embeddings=[[0.1, 0.2, 0.3]]))

    # Mock the Qdrant client
    service.qdrant_client = MagicMock()
    service.qdrant_client.search = MagicMock(return_value=[])

    return service


@pytest.mark.asyncio
async def test_retrieve_similar_chunks_basic():
    """Test basic retrieval of similar chunks"""
    service = RetrievalService()

    # Mock the dependencies
    with patch.object(service, 'generate_query_embedding', return_value=[0.1, 0.2, 0.3]), \
         patch.object(service, 'search_in_qdrant', return_value=[]):

        validation_request = ValidationRequest(
            query="test query",
            expected_sources=None,
            filters=None,
            top_k=5
        )

        results = await service.retrieve_similar_chunks(validation_request)

        assert isinstance(results, list)
        # The search_in_qdrant method was mocked to return an empty list
        assert len(results) == 0


@pytest.mark.asyncio
async def test_calculate_semantic_alignment():
    """Test calculation of semantic alignment"""
    service = RetrievalService()

    # Create mock results
    mock_results = [
        RetrievalResult(
            id="test_id_1",
            content="This is a test content for semantic alignment",
            source_url="http://example.com",
            similarity_score=0.8,
            confidence_score=0.8,
            retrieval_timestamp=MagicMock()
        )
    ]

    # Mock the Cohere client
    with patch.object(service, 'cohere_client') as mock_cohere:
        mock_cohere.embed = AsyncMock(return_value=MagicMock(embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))

        alignment_score = await service.calculate_semantic_alignment("test query", mock_results)

        # Should return a float between 0 and 1
        assert isinstance(alignment_score, float)
        assert 0.0 <= alignment_score <= 1.0


@pytest.mark.asyncio
async def test_calculate_precision_with_expected_sources():
    """Test calculation of precision with expected sources"""
    service = RetrievalService()

    # Create mock results
    mock_results = [
        RetrievalResult(
            id="result_1",
            content="Test content 1",
            source_url="http://example.com/doc1",
            similarity_score=0.8,
            confidence_score=0.8,
            retrieval_timestamp=MagicMock()
        ),
        RetrievalResult(
            id="result_2",
            content="Test content 2",
            source_url="http://example.com/doc2",
            similarity_score=0.7,
            confidence_score=0.7,
            retrieval_timestamp=MagicMock()
        )
    ]

    expected_sources = ["http://example.com/doc1", "http://example.com/doc3"]

    precision = await service.calculate_precision(mock_results, expected_sources)

    # Should return a float between 0 and 1
    assert isinstance(precision, float)
    assert 0.0 <= precision <= 1.0


@pytest.mark.asyncio
async def test_calculate_precision_without_expected_sources():
    """Test calculation of precision without expected sources"""
    service = RetrievalService()

    # Create mock results
    mock_results = [
        RetrievalResult(
            id="result_1",
            content="Test content 1",
            source_url="http://example.com/doc1",
            similarity_score=0.8,
            confidence_score=0.8,
            retrieval_timestamp=MagicMock()
        )
    ]

    precision = await service.calculate_precision(mock_results, None)

    # Without expected sources, should return 1.0 (assuming all results are relevant)
    assert precision == 1.0


def test_apply_metadata_filters():
    """Test applying metadata filters to results"""
    service = RetrievalService()

    # Create mock results
    mock_results = [
        RetrievalResult(
            id="result_1",
            content="Test content 1",
            source_url="http://example.com/doc1",
            chapter="Chapter 1",
            section="Section 1.1",
            similarity_score=0.8,
            confidence_score=0.8,
            retrieval_timestamp=MagicMock()
        ),
        RetrievalResult(
            id="result_2",
            content="Test content 2",
            source_url="http://example.com/doc2",
            chapter="Chapter 2",
            section="Section 2.1",
            similarity_score=0.7,
            confidence_score=0.7,
            retrieval_timestamp=MagicMock()
        )
    ]

    # Test filtering by chapter
    filters = {"chapter": "Chapter 1"}
    filtered_results = service.apply_metadata_filters(mock_results, filters)

    assert len(filtered_results) == 1
    assert filtered_results[0].chapter == "Chapter 1"


@pytest.mark.asyncio
async def test_retrieve_paragraph_level_chunks():
    """Test paragraph-level chunk retrieval"""
    service = RetrievalService()

    # Mock the dependencies
    with patch.object(service, 'generate_query_embedding', return_value=[0.1, 0.2, 0.3]), \
         patch.object(service, 'search_in_qdrant', return_value=[]):

        results = await service.retrieve_paragraph_level_chunks(
            query="test paragraph query",
            top_k=3
        )

        assert isinstance(results, list)


@pytest.mark.asyncio
async def test_test_paragraph_level_precision():
    """Test paragraph-level precision testing"""
    service = RetrievalService()

    # Mock the dependencies
    with patch.object(service, 'retrieve_paragraph_level_chunks', return_value=[]):

        results = await service.test_paragraph_level_precision(
            detailed_query="test detailed query",
            expected_paragraph_content="expected content",
            top_k=3
        )

        assert isinstance(results, dict)
        assert "query" in results
        assert results["query"] == "test detailed query"


@pytest.mark.asyncio
async def test_run_manual_test_queries():
    """Test running manual test queries"""
    service = RetrievalService()

    test_queries = [
        {
            "query": "test query 1",
            "expected_sources": ["http://example.com/doc1"]
        },
        {
            "query": "test query 2",
            "expected_sources": ["http://example.com/doc2"]
        }
    ]

    # Mock the dependencies
    with patch.object(service, 'retrieve_similar_chunks', return_value=[]), \
         patch.object(service, 'calculate_precision', return_value=0.8), \
         patch.object(service, 'calculate_semantic_alignment', return_value=0.85):

        results = await service.run_manual_test_queries(test_queries)

        assert isinstance(results, dict)
        assert "test_queries_executed" in results
        assert results["test_queries_executed"] == 2


@pytest.mark.asyncio
async def test_compare_retrieved_with_source():
    """Test comparing retrieved results with source content"""
    service = RetrievalService()

    mock_results = [
        RetrievalResult(
            id="result_1",
            content="Test content for comparison",
            source_url="http://example.com/doc1",
            similarity_score=0.8,
            confidence_score=0.8,
            retrieval_timestamp=MagicMock()
        )
    ]

    # Mock the Cohere client
    with patch.object(service, 'cohere_client') as mock_cohere:
        mock_cohere.embed = AsyncMock(return_value=MagicMock(embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))

        comparison = await service.compare_retrieved_with_source(
            query="test query",
            retrieved_results=mock_results,
            expected_content="expected content"
        )

        assert isinstance(comparison, dict)
        assert "query" in comparison
        assert comparison["query"] == "test query"


if __name__ == "__main__":
    pytest.main([__file__])