"""
API router for subagents in the Reusable Intelligence System
Handles API endpoints for all subagent interactions
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
import logging
import asyncio

from ..models.api_responses import (
    SubagentResponse, ChapterDraftResponse, ReviewResultsResponse,
    SummaryResultResponse, ErrorResponse
)
from ..subagents.chapter_author import ChapterAuthor
from ..subagents.technical_reviewer import TechnicalReviewer
from ..subagents.summarizer import Summarizer


# Set up logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1/subagents", tags=["subagents"])

# Initialize subagents
chapter_author = ChapterAuthor()
technical_reviewer = TechnicalReviewer()
summarizer = Summarizer()


@router.post("/chapter-writer", response_model=ChapterDraftResponse)
async def chapter_writer_endpoint(input_data: Dict[str, Any]):
    """
    Endpoint for the Chapter Author subagent
    Generates structured chapter drafts based on topic and requirements
    """
    try:
        logger.info(f"Received request to chapter writer: {input_data.get('topic', 'Unknown topic')[:50]}...")

        # Validate required inputs
        if "topic" not in input_data or "requirements" not in input_data:
            raise HTTPException(status_code=400, detail="Missing required fields: topic and requirements")

        # Execute the chapter author subagent
        result = await chapter_author.execute(input_data)

        logger.info(f"Chapter writer completed successfully for topic: {input_data['topic'][:50]}...")
        return ChapterDraftResponse(success=True, chapter_draft=result, message="Chapter draft generated successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chapter writer endpoint: {str(e)}")
        error_response = ErrorResponse(f"Chapter writer error: {str(e)}", "CHAPTER_WRITER_ERROR")
        return ChapterDraftResponse(success=False, chapter_draft={}, message=str(e))


@router.post("/content-reviewer", response_model=ReviewResultsResponse)
async def content_reviewer_endpoint(input_data: Dict[str, Any]):
    """
    Endpoint for the Technical Reviewer subagent
    Reviews content for technical accuracy and consistency
    """
    try:
        logger.info(f"Received request to content reviewer for content: {input_data.get('content', '')[:50]}...")

        # Validate required inputs
        if "content" not in input_data:
            raise HTTPException(status_code=400, detail="Missing required field: content")

        # Execute the technical reviewer subagent
        result = await technical_reviewer.execute(input_data)

        logger.info("Content reviewer completed successfully")
        return ReviewResultsResponse(success=True, review_results=result, message="Content review completed successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in content reviewer endpoint: {str(e)}")
        error_response = ErrorResponse(f"Content reviewer error: {str(e)}", "CONTENT_REVIEWER_ERROR")
        return ReviewResultsResponse(success=False, review_results={}, message=str(e))


@router.post("/summarizer", response_model=SummaryResultResponse)
async def summarizer_endpoint(input_data: Dict[str, Any]):
    """
    Endpoint for the Summarizer subagent
    Generates summaries and structured reasoning based on content
    """
    try:
        logger.info(f"Received request to summarizer for content: {input_data.get('content', '')[:50]}...")

        # Validate required inputs
        if "content" not in input_data:
            raise HTTPException(status_code=400, detail="Missing required field: content")

        # Execute the summarizer subagent
        result = await summarizer.execute(input_data)

        logger.info("Summarizer completed successfully")
        return SummaryResultResponse(success=True, summary_result=result, message="Summary generated successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in summarizer endpoint: {str(e)}")
        error_response = ErrorResponse(f"Summarizer error: {str(e)}", "SUMMARIZER_ERROR")
        return SummaryResultResponse(success=False, summary_result={}, message=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint for subagent services
    """
    try:
        # Check if all subagents are initialized
        subagents_available = [
            chapter_author is not None,
            technical_reviewer is not None,
            summarizer is not None
        ]

        all_available = all(subagents_available)

        return {
            "status": "healthy" if all_available else "unhealthy",
            "subagents": {
                "chapter_author": chapter_author is not None,
                "technical_reviewer": technical_reviewer is not None,
                "summarizer": summarizer is not None
            },
            "timestamp": str(datetime.now())
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": str(datetime.now())
        }


# Import datetime for health check
from datetime import datetime