"""
Pytest configuration for RAG Agent tests
"""
import pytest
import sys
import os

# Add the backend directory to the path so we can import from rag_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules for fixtures
from unittest.mock import Mock
from rag_agent.services.rag_agent_service import RAGAgentService
from rag_agent.tools.retrieval_tool import RetrievalTool


@pytest.fixture(scope="session")
def agent_service_mock():
    """Session-scoped mock agent service for testing"""
    service = RAGAgentService()
    service.client = Mock()
    service.client.chat.completions.create = Mock(return_value=Mock(
        choices=[Mock(message=Mock(content="Mocked response for testing"))]
    ))
    service.retrieval_tool = Mock(spec=RetrievalTool)
    service.retrieval_tool.search = Mock(return_value=[])
    return service