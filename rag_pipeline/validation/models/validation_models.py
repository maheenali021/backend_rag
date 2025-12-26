"""
Data models for the retrieval validation system
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class RetrievalQuery(BaseModel):
    """
    Represents a user's information request that triggers the retrieval process,
    containing the text query and optional metadata constraints
    """
    query_text: str = Field(..., description="The user's search query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters (chapter, section, etc.)")
    top_k: int = Field(default=5, description="Number of results to retrieve (default: 5)")
    query_embedding: Optional[List[float]] = Field(None, description="Vector representation of query")


class RetrievalResult(BaseModel):
    """
    Contains the retrieved document chunks with relevance scores, source information, and confidence metrics
    """
    id: str = Field(..., description="Unique identifier for the result")
    content: str = Field(..., description="Retrieved document chunk content")
    source_url: str = Field(..., description="URL of the source document")
    chapter: Optional[str] = Field(None, description="Chapter title from metadata")
    section: Optional[str] = Field(None, description="Section title from metadata")
    similarity_score: float = Field(..., description="Similarity score (0-1)", ge=0.0, le=1.0)
    confidence_score: float = Field(..., description="Confidence in relevance (0-1)", ge=0.0, le=1.0)
    retrieval_timestamp: datetime = Field(default_factory=datetime.now, description="When retrieved")


class ValidationRequest(BaseModel):
    """
    Request model for validating retrieval quality
    """
    query: str = Field(..., description="Original user query")
    expected_sources: Optional[List[str]] = Field(None, description="Expected source documents (optional)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters to apply")
    top_k: int = Field(default=5, description="Number of results to retrieve (default: 5)")


class ValidationResponse(BaseModel):
    """
    Response model for retrieval validation results
    """
    query: str = Field(..., description="Original query")
    retrieved_results: List[RetrievalResult] = Field(..., description="Retrieved content chunks")
    semantic_alignment_score: float = Field(..., description="Score of alignment with intent", ge=0.0, le=1.0)
    precision_metric: float = Field(..., description="Precision of results", ge=0.0, le=1.0)
    latency_ms: float = Field(..., description="Time taken for retrieval in milliseconds")
    validation_passed: bool = Field(..., description="Whether validation criteria were met")
    validation_timestamp: datetime = Field(default_factory=datetime.now, description="When validation was performed")


class RetrievalQualityMetrics(BaseModel):
    """
    Quality metrics for retrieval performance
    """
    precision: float = Field(..., description="Precision of retrieval results", ge=0.0, le=1.0)
    recall: float = Field(..., description="Recall of retrieval results", ge=0.0, le=1.0)
    f1_score: float = Field(..., description="F1 score of retrieval results", ge=0.0, le=1.0)
    semantic_alignment: float = Field(..., description="Semantic alignment score", ge=0.0, le=1.0)
    latency_p50: float = Field(..., description="50th percentile latency in milliseconds")
    latency_p95: float = Field(..., description="95th percentile latency in milliseconds")
    latency_p99: float = Field(..., description="99th percentile latency in milliseconds")
    success_rate: float = Field(..., description="Success rate of retrieval operations", ge=0.0, le=1.0)


class ValidationResult(BaseModel):
    """
    Result of a single validation operation
    """
    query: str = Field(..., description="The query that was validated")
    expected_sources: Optional[List[str]] = Field(None, description="Expected source documents")
    retrieved_sources: List[str] = Field(..., description="Actually retrieved source documents")
    quality_metrics: RetrievalQualityMetrics = Field(..., description="Quality metrics for this validation")
    is_valid: bool = Field(..., description="Whether the retrieval quality is acceptable")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional validation details")


class RetrievalFilterType(str, Enum):
    """
    Types of metadata filters available for retrieval
    """
    CHAPTER = "chapter"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    PAGE = "page"
    DOCUMENT = "document"


class MetadataFilter(BaseModel):
    """
    Parameters that constrain retrieval to specific book sections
    """
    filter_type: RetrievalFilterType = Field(..., description="Type of filter to apply")
    filter_value: str = Field(..., description="Value to filter by")
    case_sensitive: bool = Field(default=False, description="Whether the filter is case sensitive")