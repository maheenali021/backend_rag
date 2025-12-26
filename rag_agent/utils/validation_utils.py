"""
Validation utilities for the RAG Agent system
Contains validation functions for inputs, responses, and content checking
"""
import re
from typing import List, Dict, Any, Optional
from ..models.agent_models import RetrievedChunk, AgentResponse
from ..config import Config


class ValidationLogger:
    """Logger for validation events"""

    @staticmethod
    def log_validation_event(event_type: str, message: str, details: Dict[str, Any] = None):
        """Log a validation event"""
        import logging
        logger = logging.getLogger(__name__)
        log_data = {"event_type": event_type, "message": message}
        if details:
            log_data["details"] = details
        logger.info(f"VALIDATION: {log_data}")


class ContentValidator:
    """Validator for content and responses"""

    @staticmethod
    def validate_query(query: str) -> bool:
        """Validate that a query is appropriate for the agent"""
        if not query or len(query.strip()) < 3:
            return False

        # Check for potentially harmful patterns
        harmful_patterns = [
            r"(?i)system|prompt|instruction",  # Attempts to access system instructions
            r"(?i)ignore|disregard|bypass",    # Attempts to bypass safety
        ]

        for pattern in harmful_patterns:
            if re.search(pattern, query):
                return False

        return True

    @staticmethod
    def validate_retrieved_chunks(chunks: List[RetrievedChunk]) -> bool:
        """Validate that retrieved chunks are appropriate"""
        if not chunks:
            return True  # Empty is valid (no results)

        for chunk in chunks:
            if not chunk.content or len(chunk.content.strip()) == 0:
                return False
            if chunk.similarity_score < 0 or chunk.similarity_score > 1:
                return False
            if chunk.confidence_score < 0 or chunk.confidence_score > 1:
                return False

        return True

    @staticmethod
    def validate_response_grounding(response: str, retrieved_chunks: List[RetrievedChunk]) -> bool:
        """Validate that a response is grounded in the retrieved content"""
        # Handle case where response might be a Mock object during testing
        try:
            # Check if response is a Mock object or similar (during testing)
            if hasattr(response, 'return_value') or hasattr(response, 'side_effect') or str(type(response)) == "<class 'unittest.mock.Mock'>":
                # This is a mock, convert to string for testing
                response_str = str(response)
            else:
                response_str = response
        except:
            response_str = str(response)

        if not retrieved_chunks:
            # If no chunks were retrieved, the response should acknowledge this
            return "no relevant information" in response_str.lower() or "not found" in response_str.lower()

        # Check if response content is related to retrieved content
        response_lower = response_str.lower()
        for chunk in retrieved_chunks:
            chunk_lower = chunk.content.lower()
            # Check if significant portions of the response align with retrieved content
            if len(chunk_lower) > 50:  # Only check substantial chunks
                # Look for content overlap
                chunk_words = set(chunk_lower.split()[:20])  # First 20 words for efficiency
                response_words = set(response_lower.split())
                overlap = len(chunk_words.intersection(response_words))

                # If there's some overlap, consider it grounded
                if overlap > 2:  # At least 3 words in common
                    return True

        # If no substantial overlap found, check if it's acknowledging lack of info
        if "not found" in response_lower or "no relevant" in response_lower:
            return True

        return False


class HallucinationDetector:
    """Detects and prevents hallucinations in agent responses"""

    @staticmethod
    def detect_hallucinations(response: str, retrieved_chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        """Detect potential hallucinations in a response"""
        issues = []
        confidence_score = 1.0  # Start with high confidence

        # Handle case where response might be a Mock object during testing
        try:
            # Check if response is a Mock object or similar (during testing)
            if hasattr(response, 'return_value') or hasattr(response, 'side_effect') or str(type(response)) == "<class 'unittest.mock.Mock'>":
                # This is a mock, convert to string for testing
                response_str = str(response)
            else:
                response_str = response
        except:
            response_str = str(response)

        # Check if response contains claims not supported by retrieved content
        response_lower = response_str.lower()

        # Look for confidence indicators in the response
        uncertain_indicators = [
            "i think", "i believe", "possibly", "might be", "could be",
            "perhaps", "seems like", "appears to", "not sure", "uncertain"
        ]

        uncertain_found = any(indicator in response_lower for indicator in uncertain_indicators)

        # Check grounding in retrieved content
        grounded = ContentValidator.validate_response_grounding(response_str, retrieved_chunks)

        if not grounded and not uncertain_found:
            issues.append("Response may contain information not supported by retrieved content")
            confidence_score *= 0.5  # Reduce confidence

        # Check for specific claims that should be in retrieved content
        if retrieved_chunks:
            # Look for specific numbers, dates, or technical terms that should be verified
            import re
            numbers_in_response = re.findall(r'\b\d{4}\b', response_str)  # Years
            for number in numbers_in_response:
                # Check if this number appears in retrieved content
                found_in_chunks = any(number in chunk.content for chunk in retrieved_chunks)
                if not found_in_chunks:
                    issues.append(f"Number '{number}' found in response but not in retrieved content")
                    confidence_score *= 0.8

        return {
            "has_potential_hallucinations": len(issues) > 0,
            "issues": issues,
            "confidence_adjustment": confidence_score
        }


class ResponseValidator:
    """Validator for agent responses"""

    @staticmethod
    def validate_agent_response(response: AgentResponse) -> Dict[str, Any]:
        """Validate an agent response for quality and safety"""
        issues = []

        # Validate basic response structure
        if not response.response or len(response.response.strip()) == 0:
            issues.append("Response is empty")

        if not response.query:
            issues.append("Original query is missing")

        if response.confidence_score < 0 or response.confidence_score > 1:
            issues.append("Confidence score out of range")

        # Validate retrieved chunks
        if not ContentValidator.validate_retrieved_chunks(response.retrieved_chunks):
            issues.append("Retrieved chunks validation failed")

        # Check grounding
        grounded = ContentValidator.validate_response_grounding(
            response.response,
            response.retrieved_chunks
        )

        if not grounded:
            issues.append("Response is not adequately grounded in retrieved content")

        # Detect hallucinations
        hallucination_check = HallucinationDetector.detect_hallucinations(
            response.response,
            response.retrieved_chunks
        )

        return {
            "is_valid": len(issues) == 0 and not hallucination_check["has_potential_hallucinations"],
            "issues": issues,
            "hallucination_check": hallucination_check,
            "grounding_score": 1.0 if grounded else 0.0
        }


class QueryValidator:
    """Validator for user queries"""

    @staticmethod
    def validate_agent_request(request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an agent request"""
        issues = []

        # Check required fields
        if not request.get("query"):
            issues.append("Query is required")

        # Validate query content
        if request.get("query") and not ContentValidator.validate_query(request["query"]):
            issues.append("Query contains invalid content or patterns")

        # Validate query type
        query_type = request.get("query_type", "general")
        valid_types = ["general", "chapter_specific", "user_context"]
        if query_type not in valid_types:
            issues.append(f"Invalid query type. Must be one of {valid_types}")

        # Validate filters
        filters = request.get("filters")
        if filters and not isinstance(filters, dict):
            issues.append("Filters must be a dictionary")

        # Validate top_k
        top_k = request.get("top_k", 5)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            issues.append("top_k must be an integer between 1 and 20")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues
        }