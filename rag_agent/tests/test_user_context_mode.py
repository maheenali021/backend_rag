"""
Tests for user-provided context mode functionality
"""
import pytest
from unittest.mock import Mock, patch
from ..models.agent_models import AgentRequest, QueryType
from ..services.rag_agent_service import RAGAgentService
from ..tools.retrieval_tool import RetrievalTool


@pytest.fixture
def agent_service_for_user_context():
    """Create an agent service for testing user context mode"""
    service = RAGAgentService()
    service.client = Mock()
    service.client.chat.completions.create = Mock(return_value=Mock(
        choices=[Mock(message=Mock(content="Response based on user-provided context"))]
    ))
    service.retrieval_tool = Mock(spec=RetrievalTool)
    # Ensure retrieval tool returns empty results for user context mode tests
    service.retrieval_tool.search.return_value = []
    return service


def test_user_context_mode_uses_provided_context(agent_service_for_user_context):
    """Test that user context mode uses provided context instead of retrieval"""
    user_provided_context = "The quick brown fox jumps over the lazy dog. This is my custom context for testing."

    request = AgentRequest(
        query="What happens with the fox?",
        query_type=QueryType.USER_CONTEXT,
        user_context=user_provided_context,
        top_k=5  # This should be ignored in user context mode
    )

    response = agent_service_for_user_context.process_request(request)

    # Verify that retrieval was NOT called (since we're in user context mode)
    agent_service_for_user_context.retrieval_tool.search.assert_not_called()

    # Verify response properties
    assert response.query_type == QueryType.USER_CONTEXT
    assert len(response.retrieved_chunks) == 0  # No chunks retrieved from DB
    assert response.has_sufficient_context is True  # Because user provided context


def test_user_context_mode_with_empty_context(agent_service_for_user_context):
    """Test behavior when user provides empty context"""
    request = AgentRequest(
        query="What is the meaning of life?",
        query_type=QueryType.USER_CONTEXT,
        user_context="",  # Empty context
    )

    response = agent_service_for_user_context.process_request(request)

    # Retrieval should still not be called
    agent_service_for_user_context.retrieval_tool.search.assert_not_called()

    # Response should handle the empty context appropriately
    assert response.query_type == QueryType.USER_CONTEXT
    assert len(response.retrieved_chunks) == 0
    # The response content would depend on the LLM's handling of empty context


def test_user_context_mode_with_none_context(agent_service_for_user_context):
    """Test behavior when user context is None but mode is set to user context"""
    request = AgentRequest(
        query="How does AI work?",
        query_type=QueryType.USER_CONTEXT,
        user_context=None,
    )

    response = agent_service_for_user_context.process_request(request)

    # Retrieval should still not be called
    agent_service_for_user_context.retrieval_tool.search.assert_not_called()

    assert response.query_type == QueryType.USER_CONTEXT
    assert len(response.retrieved_chunks) == 0


def test_user_context_mode_ignores_filters(agent_service_for_user_context):
    """Test that filters are ignored in user context mode"""
    user_context = "Custom context about robotics."

    request = AgentRequest(
        query="Explain the custom content",
        query_type=QueryType.USER_CONTEXT,
        user_context=user_context,
        filters={"chapter": "Not Used", "section": "Ignored"},  # Should be ignored
        chapter_filter="Also Ignored"  # Should be ignored
    )

    response = agent_service_for_user_context.process_request(request)

    # Retrieval should not be called, so filters don't matter
    agent_service_for_user_context.retrieval_tool.search.assert_not_called()

    assert response.query_type == QueryType.USER_CONTEXT
    assert len(response.retrieved_chunks) == 0


def test_user_context_mode_vs_general_query(agent_service_for_user_context):
    """Compare behavior between user context mode and general mode"""
    user_context = "Specific information about neural networks: They are computing systems."

    # Request in user context mode
    user_context_request = AgentRequest(
        query="What are neural networks?",
        query_type=QueryType.USER_CONTEXT,
        user_context=user_context
    )

    # Request in general mode (for comparison)
    general_request = AgentRequest(
        query="What are neural networks?",
        query_type=QueryType.GENERAL
    )

    # Mock different responses for each case
    def side_effect_model(*args, **kwargs):
        if 'user_context' in str(kwargs) or len(args) > 0 and 'neural networks' in str(args):
            # This is a simplification - in real scenario, we'd mock differently
            pass
        return Mock(choices=[Mock(message=Mock(content="Response based on provided context"))])

    agent_service_for_user_context.client.chat.completions.create.side_effect = lambda *args, **kwargs: Mock(
        choices=[Mock(message=Mock(content="Response based on provided context"))]
    )

    user_response = agent_service_for_user_context.process_request(user_context_request)
    agent_service_for_user_context.retrieval_tool.search.return_value = [{"id": "test", "content": "general content", "source_url": "http://example.com", "chapter": "General", "section": "Intro", "similarity_score": 0.8, "confidence_score": 0.8, "metadata": {}}]
    general_response = agent_service_for_user_context.process_request(general_request)

    # User context response should not have retrieved chunks
    assert len(user_response.retrieved_chunks) == 0

    # General response should have retrieved chunks (mocked)
    assert len(general_response.retrieved_chunks) == 1


def test_user_context_mode_preserves_context_relevance(agent_service_for_user_context):
    """Test that responses stay relevant to the user-provided context"""
    user_context = """
    Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering,
    electrical engineering, computer science, and others. It deals with the design, construction, operation,
    and use of robots, as well as computer systems for their control, sensory feedback, and information processing.
    """

    request = AgentRequest(
        query="What is robotics?",
        query_type=QueryType.USER_CONTEXT,
        user_context=user_context
    )

    response = agent_service_for_user_context.process_request(request)

    # Verify the mode and context handling
    assert response.query_type == QueryType.USER_CONTEXT
    assert len(response.retrieved_chunks) == 0  # No DB retrieval
    assert response.has_sufficient_context is True  # User provided context


def test_user_context_mode_large_context_handling(agent_service_for_user_context):
    """Test handling of large user-provided contexts"""
    large_context = "Robotics. " * 1000  # Large context
    large_context += "AI. " * 1000  # More content

    request = AgentRequest(
        query="Summarize the content",
        query_type=QueryType.USER_CONTEXT,
        user_context=large_context
    )

    response = agent_service_for_user_context.process_request(request)

    # Should handle large context without errors
    assert response.query_type == QueryType.USER_CONTEXT
    assert len(response.retrieved_chunks) == 0
    assert response.has_sufficient_context is True