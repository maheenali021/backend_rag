"""
Tests specifically for chapter-specific query functionality
"""
import pytest
from unittest.mock import Mock, patch
from ..models.agent_models import AgentRequest, QueryType
from ..services.rag_agent_service import RAGAgentService
from ..tools.retrieval_tool import RetrievalTool


@pytest.fixture
def agent_service_with_mock_retrieval():
    """Create an agent service with mocked retrieval functionality"""
    service = RAGAgentService()
    service.client = Mock()
    service.client.chat.completions.create = Mock(return_value=Mock(
        choices=[Mock(message=Mock(content="Chapter-specific response based on filtered results"))]
    ))
    service.retrieval_tool = Mock(spec=RetrievalTool)
    return service


def test_chapter_specific_query_calls_retrieval_with_filters(agent_service_with_mock_retrieval):
    """Test that chapter-specific queries properly filter the retrieval"""
    # Mock retrieval results for a specific chapter
    mock_results = [
        {
            "id": "ch3_sec1",
            "content": "This content is from Chapter 3, Section 1 about neural networks.",
            "source_url": "http://example.com/chapter3",
            "chapter": "Neural Networks",
            "section": "Introduction",
            "similarity_score": 0.90,
            "confidence_score": 0.88,
            "metadata": {}
        },
        {
            "id": "ch3_sec2",
            "content": "This content is from Chapter 3, Section 2 about deep learning.",
            "source_url": "http://example.com/chapter3#section2",
            "chapter": "Neural Networks",
            "section": "Deep Learning",
            "similarity_score": 0.85,
            "confidence_score": 0.82,
            "metadata": {}
        }
    ]
    agent_service_with_mock_retrieval.retrieval_tool.search.return_value = mock_results

    # Create a chapter-specific request
    request = AgentRequest(
        query="Explain neural networks",
        query_type=QueryType.CHAPTER_SPECIFIC,
        chapter_filter="Neural Networks",
        top_k=5
    )

    # Process the request
    response = agent_service_with_mock_retrieval.process_request(request)

    # Verify that the retrieval tool was called with the correct filters
    agent_service_with_mock_retrieval.retrieval_tool.search.assert_called_once_with(
        "Explain neural networks",
        filters={"chapter": "Neural Networks"},
        top_k=5
    )

    # Verify response properties
    assert response.query_type == QueryType.CHAPTER_SPECIFIC
    assert len(response.retrieved_chunks) == 2
    assert all(chunk.chapter == "Neural Networks" for chunk in response.retrieved_chunks)
    assert response.has_sufficient_context is True


def test_chapter_specific_query_with_multiple_filters(agent_service_with_mock_retrieval):
    """Test chapter-specific queries with additional filters"""
    # Mock retrieval results
    mock_results = [
        {
            "id": "ch5_sec3",
            "content": "Advanced robotics content from chapter 5.",
            "source_url": "http://example.com/chapter5",
            "chapter": "Advanced Robotics",
            "section": "Path Planning",
            "similarity_score": 0.95,
            "confidence_score": 0.92,
            "metadata": {"difficulty": "advanced"}
        }
    ]
    agent_service_with_mock_retrieval.retrieval_tool.search.return_value = mock_results

    # Create a request with chapter filter and additional filters
    request = AgentRequest(
        query="How does path planning work?",
        query_type=QueryType.CHAPTER_SPECIFIC,
        chapter_filter="Advanced Robotics",
        filters={"difficulty": "advanced"},
        top_k=3
    )

    # Process the request
    response = agent_service_with_mock_retrieval.process_request(request)

    # Verify that the retrieval tool was called with combined filters
    agent_service_with_mock_retrieval.retrieval_tool.search.assert_called_once_with(
        "How does path planning work?",
        filters={"chapter": "Advanced Robotics", "difficulty": "advanced"},
        top_k=3
    )

    assert len(response.retrieved_chunks) == 1
    assert response.retrieved_chunks[0].chapter == "Advanced Robotics"
    assert response.confidence_score > 0.5


def test_chapter_specific_query_no_results(agent_service_with_mock_retrieval):
    """Test chapter-specific queries when no results are found in the specified chapter"""
    # Mock empty retrieval results
    agent_service_with_mock_retrieval.retrieval_tool.search.return_value = []

    # Create a chapter-specific request
    request = AgentRequest(
        query="Explain quantum computing",
        query_type=QueryType.CHAPTER_SPECIFIC,
        chapter_filter="Non-Existent Chapter",
        top_k=5
    )

    # Process the request
    response = agent_service_with_mock_retrieval.process_request(request)

    # Verify that the retrieval was attempted with the chapter filter
    agent_service_with_mock_retrieval.retrieval_tool.search.assert_called_once_with(
        "Explain quantum computing",
        filters={"chapter": "Non-Existent Chapter"},
        top_k=5
    )

    # Response should indicate lack of relevant information
    assert response.has_sufficient_context is False
    assert len(response.retrieved_chunks) == 0
    # The response text would depend on what the LLM generates, but it should acknowledge lack of info


def test_chapter_specific_query_accuracy(agent_service_with_mock_retrieval):
    """Test that chapter-specific queries return results primarily from the specified chapter"""
    # Mock results where most results are from the requested chapter
    mock_results = [
        {
            "id": "ch2_sec1",
            "content": "Content from Chapter 2, Section 1.",
            "source_url": "http://example.com/chapter2/sec1",
            "chapter": "Machine Learning Basics",
            "section": "Introduction",
            "similarity_score": 0.88,
            "confidence_score": 0.85,
            "metadata": {}
        },
        {
            "id": "ch2_sec2",
            "content": "Content from Chapter 2, Section 2.",
            "source_url": "http://example.com/chapter2/sec2",
            "chapter": "Machine Learning Basics",
            "section": "Algorithms",
            "similarity_score": 0.82,
            "confidence_score": 0.80,
            "metadata": {}
        },
        {
            "id": "ch2_sec3",
            "content": "More content from Chapter 2.",
            "source_url": "http://example.com/chapter2/sec3",
            "chapter": "Machine Learning Basics",
            "section": "Applications",
            "similarity_score": 0.78,
            "confidence_score": 0.75,
            "metadata": {}
        }
    ]
    agent_service_with_mock_retrieval.retrieval_tool.search.return_value = mock_results

    request = AgentRequest(
        query="What are machine learning basics?",
        query_type=QueryType.CHAPTER_SPECIFIC,
        chapter_filter="Machine Learning Basics",
        top_k=5
    )

    response = agent_service_with_mock_retrieval.process_request(request)

    # Verify all results are from the requested chapter
    assert all(chunk.chapter == "Machine Learning Basics" for chunk in response.retrieved_chunks)
    assert len(response.retrieved_chunks) == 3
    assert response.query_type == QueryType.CHAPTER_SPECIFIC


def test_chapter_specific_query_fallback_to_general(agent_service_with_mock_retrieval):
    """Test behavior when chapter-specific query yields few results"""
    # Mock results with low similarity scores
    mock_results = [
        {
            "id": "ch4_sec1",
            "content": "Loosely related content from Chapter 4.",
            "source_url": "http://example.com/chapter4",
            "chapter": "Computer Vision",
            "section": "Introduction",
            "similarity_score": 0.30,  # Low similarity
            "confidence_score": 0.25,
            "metadata": {}
        }
    ]
    agent_service_with_mock_retrieval.retrieval_tool.search.return_value = mock_results

    request = AgentRequest(
        query="Explain computer vision techniques",
        query_type=QueryType.CHAPTER_SPECIFIC,
        chapter_filter="Computer Vision",
        top_k=5
    )

    response = agent_service_with_mock_retrieval.process_request(request)

    # Verify the chapter filter was still applied in the retrieval call
    agent_service_with_mock_retrieval.retrieval_tool.search.assert_called_once_with(
        "Explain computer vision techniques",
        filters={"chapter": "Computer Vision"},
        top_k=5
    )

    # Results should still be from the specified chapter, even if low quality
    assert response.retrieved_chunks[0].chapter == "Computer Vision"
    assert response.confidence_score < 0.5  # Low confidence due to low similarity scores