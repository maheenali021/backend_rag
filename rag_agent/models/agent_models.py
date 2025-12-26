"""
Agent models for the RAG Agent system
Contains data models for requests, responses, and conversation sessions
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class QueryType(str, Enum):
    """Types of queries supported by the agent"""
    GENERAL = "general"
    CHAPTER_SPECIFIC = "chapter_specific"
    USER_CONTEXT = "user_context"


class RetrievedChunk(BaseModel):
    """Model for a retrieved content chunk with metadata and source attribution"""
    id: str = Field(..., description="Unique identifier for the retrieved chunk")
    content: str = Field(..., description="The retrieved content text")
    source_url: str = Field(..., description="URL of the source document")
    chapter: Optional[str] = Field(None, description="Chapter title from metadata")
    section: Optional[str] = Field(None, description="Section title from metadata")
    similarity_score: float = Field(..., description="Similarity score (0-1)", ge=0.0, le=1.0)
    confidence_score: float = Field(..., description="Confidence in relevance (0-1)", ge=0.0, le=1.0)
    retrieval_timestamp: datetime = Field(default_factory=datetime.now, description="When retrieved")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AgentRequest(BaseModel):
    """Model for requests to the agent"""
    query: str = Field(..., description="User query text", min_length=1)
    query_type: QueryType = Field(QueryType.GENERAL, description="Type of query")
    chapter_filter: Optional[str] = Field(None, description="Chapter filter for chapter-specific queries")
    user_context: Optional[str] = Field(None, description="User-provided context for user context mode")
    conversation_id: Optional[str] = Field(None, description="ID of the conversation session")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional metadata filters")
    top_k: int = Field(5, description="Number of results to retrieve", ge=1, le=20)


class AgentResponse(BaseModel):
    """Model for responses from the agent"""
    response: str = Field(..., description="The agent's response to the user query")
    query: str = Field(..., description="Original query that generated this response")
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list, description="Retrieved content chunks used to generate response")
    source_attribution: List[str] = Field(default_factory=list, description="Sources used in the response")
    confidence_score: float = Field(..., description="Overall confidence in the response (0-1)", ge=0.0, le=1.0)
    query_type: QueryType = Field(..., description="Type of query that was processed")
    conversation_id: str = Field(..., description="ID of the conversation session")
    response_timestamp: datetime = Field(default_factory=datetime.now, description="When response was generated")
    has_sufficient_context: bool = Field(True, description="Whether sufficient context was available to answer")
    hallucination_prevention_applied: bool = Field(False, description="Whether hallucination prevention measures were applied")


class ConversationSession(BaseModel):
    """Model for conversation session state"""
    id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="When session was created")
    last_interaction: datetime = Field(default_factory=datetime.now, description="When last interaction occurred")
    history: List[Dict[str, str]] = Field(default_factory=list, description="Conversation history")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional session metadata")
    active: bool = Field(True, description="Whether the session is currently active")

    def add_interaction(self, user_query: str, agent_response: str):
        """Add a user-agent interaction to the conversation history"""
        self.history.append({
            "user": user_query,
            "agent": agent_response,
            "timestamp": datetime.now().isoformat()
        })
        self.last_interaction = datetime.now()