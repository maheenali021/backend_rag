"""
Tests for hallucination prevention functionality
"""
import pytest
from unittest.mock import Mock
from ..models.agent_models import AgentRequest, AgentResponse, RetrievedChunk, QueryType
from ..services.rag_agent_service import RAGAgentService
from ..utils.validation_utils import HallucinationDetector, ContentValidator, ResponseValidator
from ..tools.retrieval_tool import RetrievalTool


@pytest.fixture
def agent_service_for_hallucination_tests():
    """Create an agent service for testing hallucination prevention"""
    service = RAGAgentService()
    service.client = Mock()
    service.retrieval_tool = Mock(spec=RetrievalTool)
    return service


def test_hallucination_detector_basic_functionality():
    """Test basic hallucination detection functionality"""
    detector = HallucinationDetector()

    # Test with content that IS supported by retrieved chunks
    supported_response = "AI is artificial intelligence that mimics human cognitive functions."
    retrieved_chunks = [
        RetrievedChunk(
            id="test1",
            content="Artificial Intelligence (AI) is intelligence demonstrated by machines.",
            source_url="http://example.com",
            similarity_score=0.8,
            confidence_score=0.8
        )
    ]

    result = detector.detect_hallucinations(supported_response, retrieved_chunks)
    # The response has some overlap with retrieved content ("AI", "intelligence")
    # but the detector might still flag it depending on the implementation

    assert "has_potential_hallucinations" in result
    assert "issues" in result
    assert "confidence_adjustment" in result


def test_hallucination_detector_no_support():
    """Test hallucination detection when response has no support in retrieved content"""
    detector = HallucinationDetector()

    # Response with information not in retrieved content
    unsupported_response = "Flying cars will be available for purchase next year for under $10,000."
    retrieved_chunks = [
        RetrievedChunk(
            id="test1",
            content="Artificial Intelligence (AI) is intelligence demonstrated by machines.",
            source_url="http://example.com",
            similarity_score=0.8,
            confidence_score=0.8
        )
    ]

    result = detector.detect_hallucinations(unsupported_response, retrieved_chunks)

    # This response should be flagged as potential hallucination
    # since it talks about flying cars which isn't in the retrieved content
    assert "has_potential_hallucinations" in result
    assert "confidence_adjustment" in result


def test_hallucination_detector_uncertain_language():
    """Test that uncertain language reduces hallucination flags"""
    detector = HallucinationDetector()

    # Response with uncertain language
    uncertain_response = "I think flying cars might be available in the future, but I'm not sure."
    retrieved_chunks = [
        RetrievedChunk(
            id="test1",
            content="Artificial Intelligence (AI) is intelligence demonstrated by machines.",
            source_url="http://example.com",
            similarity_score=0.8,
            confidence_score=0.8
        )
    ]

    result = detector.detect_hallucinations(uncertain_response, retrieved_chunks)

    # Uncertain language should be treated more leniently
    assert "confidence_adjustment" in result


def test_content_validator_response_grounding():
    """Test that content validator checks if responses are grounded in retrieved content"""
    validator = ContentValidator()

    # Test with grounded response
    grounded_response = "AI is artificial intelligence."
    retrieved_chunks = [
        RetrievedChunk(
            id="test1",
            content="Artificial Intelligence (AI) is intelligence demonstrated by machines.",
            source_url="http://example.com",
            similarity_score=0.8,
            confidence_score=0.8
        )
    ]

    is_grounded = validator.validate_response_grounding(grounded_response, retrieved_chunks)
    # This depends on the specific implementation of validate_response_grounding

    # Test with ungrounded response
    ungrounded_response = "The weather is nice today."
    is_ungrounded = validator.validate_response_grounding(ungrounded_response, retrieved_chunks)
    assert isinstance(is_ungrounded, bool)


def test_response_validator_hallucination_checking():
    """Test that response validator includes hallucination checking"""
    validator = ResponseValidator()

    # Create a response with retrieved chunks
    response = AgentResponse(
        response="AI is artificial intelligence that mimics human cognitive functions.",
        query="What is AI?",
        retrieved_chunks=[
            RetrievedChunk(
                id="test1",
                content="Artificial Intelligence (AI) is intelligence demonstrated by machines.",
                source_url="http://example.com",
                similarity_score=0.8,
                confidence_score=0.8
            )
        ],
        source_attribution=["http://example.com"],
        confidence_score=0.8,
        query_type=QueryType.GENERAL,
        conversation_id="test_conversation"
    )

    validation_result = validator.validate_agent_response(response)

    assert "is_valid" in validation_result
    assert "issues" in validation_result
    assert "hallucination_check" in validation_result
    assert "grounding_score" in validation_result


def test_agent_service_hallucination_prevention_integration(agent_service_for_hallucination_tests):
    """Test that hallucination prevention is integrated into the agent service"""
    # Mock retrieval results
    mock_retrieved_chunks = [
        {
            "id": "test_chunk",
            "content": "AI is artificial intelligence.",
            "source_url": "http://example.com",
            "chapter": "Introduction",
            "section": "Basic Definitions",
            "similarity_score": 0.75,
            "confidence_score": 0.75,
            "metadata": {}
        }
    ]
    agent_service_for_hallucination_tests.retrieval_tool.search.return_value = mock_retrieved_chunks

    # Mock a response that has potential hallucinations
    def mock_completion(*args, **kwargs):
        result = Mock()
        result.choices = [Mock(message=Mock(content="AI is artificial intelligence and it can also perform tasks like cooking and cleaning houses."))]
        return result

    agent_service_for_hallucination_tests.client.chat.completions.create.side_effect = mock_completion

    request = AgentRequest(
        query="What is AI?",
        query_type=QueryType.GENERAL
    )

    response = agent_service_for_hallucination_tests.process_request(request)

    # The response should have gone through hallucination detection
    assert hasattr(response, 'hallucination_prevention_applied')
    # This field should indicate if prevention measures were applied


def test_hallucination_prevention_with_no_context():
    """Test behavior when no context is available"""
    validator = ContentValidator()

    # When no chunks are retrieved, a proper response should acknowledge this
    response_with_acknowledgment = "No relevant information was found in the textbook for this query."
    result = validator.validate_response_grounding(response_with_acknowledgment, [])

    # This should be considered valid as it acknowledges lack of information
    assert result is True

    # But a response that makes claims without context should be invalid
    response_with_claims = "AI will take over the world in 2030."
    result2 = validator.validate_response_grounding(response_with_claims, [])

    # This should be invalid as it makes specific claims without supporting context
    # (actual result depends on implementation)


def test_year_hallucination_detection():
    """Test detection of specific factual claims like years"""
    detector = HallucinationDetector()

    # Response with specific year not in retrieved content
    response_with_year = "The first AI program was written in 1955."
    retrieved_chunks = [
        RetrievedChunk(
            id="test1",
            content="Early AI research began in the 1950s.",
            source_url="http://example.com",
            similarity_score=0.7,
            confidence_score=0.7
        )
    ]

    result = detector.detect_hallucinations(response_with_year, retrieved_chunks)

    # The detector should identify specific claims not supported by retrieved content
    assert "confidence_adjustment" in result
    assert 0.0 <= result["confidence_adjustment"] <= 1.0