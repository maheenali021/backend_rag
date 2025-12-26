"""
Validation endpoints for the retrieval pipeline
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import time
import logging

from ..models.validation_models import ValidationRequest, ValidationResponse, RetrievalQuery, RetrievalResult
from ..services.retrieval_service import retrieval_service

# Set up logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1/retrieval", tags=["retrieval-validation"])

@router.post("/validate", response_model=ValidationResponse)
async def validate_retrieval_quality(request: ValidationRequest):
    """
    Validates retrieval quality for a given query
    """
    start_time = time.time()

    try:
        # Perform retrieval
        results = await retrieval_service.retrieve_similar_chunks(request)

        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000

        # Calculate semantic alignment score
        semantic_alignment_score = await retrieval_service.calculate_semantic_alignment(request.query, results)

        # Calculate precision metric
        precision_metric = await retrieval_service.calculate_precision(results, request.expected_sources)

        # Determine if validation passed based on success criteria
        validation_passed = (
            precision_metric >= 0.70 and  # At least 70% precision
            latency_ms <= 1500  # Under 1.5 seconds
        )

        response = ValidationResponse(
            query=request.query,
            retrieved_results=results,
            semantic_alignment_score=semantic_alignment_score,
            precision_metric=precision_metric,
            latency_ms=latency_ms,
            validation_passed=validation_passed
        )

        logger.info(f"Retrieval validation completed for query: '{request.query[:50]}...', "
                   f"latency: {latency_ms:.2f}ms, results: {len(results)}, "
                   f"filters: {request.filters}")

        return response

    except Exception as e:
        logger.error(f"Error during retrieval validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retrieval validation failed: {str(e)}")


@router.get("/test-interface")
async def test_retrieval_interface(
    q: str = Query(..., description="Query text"),
    k: Optional[int] = Query(5, description="Number of results to return"),
    chapter: Optional[str] = Query(None, description="Chapter filter"),
    section: Optional[str] = Query(None, description="Section filter")
):
    """
    Debug endpoint for manual testing with human-readable output
    """
    try:
        # Create a retrieval query with filters
        filters = {}
        if chapter:
            filters["chapter"] = chapter
        if section:
            filters["section"] = section

        query_obj = RetrievalQuery(
            query_text=q,
            filters=filters,
            top_k=k
        )

        # Perform retrieval
        results = await retrieval_service.retrieve_similar_chunks_from_query(query_obj)

        # Format results for human-readable output
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                "rank": i,
                "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                "full_content": result.content,  # Include full content for detailed inspection
                "source_url": result.source_url,
                "chapter": result.chapter,
                "section": result.section,
                "similarity_score": result.similarity_score,
                "confidence_score": result.confidence_score,
                "id": result.id,
                "retrieval_timestamp": result.retrieval_timestamp.isoformat()
            })

        html_content = f"""
        <html>
        <head>
            <title>Retrieval Test Interface - Detailed Inspection</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .result {{ border: 1px solid #ccc; margin: 10px 0; padding: 10px; }}
                .content {{ background-color: #f9f9f9; padding: 5px; margin: 5px 0; }}
                .meta {{ color: #666; font-size: 0.9em; }}
                .query-info {{ background-color: #e6f3ff; padding: 10px; margin: 10px 0; }}
                .detailed-view {{ display: none; }}
                .toggle-btn {{ background: #007bff; color: white; border: none; padding: 5px 10px; cursor: pointer; }}
                .highlight {{ background-color: yellow; }}
            </style>
            <script>
                function toggleDetail(elementId) {{
                    const element = document.getElementById(elementId);
                    element.style.display = element.style.display === 'none' ? 'block' : 'none';
                }}
            </script>
        </head>
        <body>
            <h1>Retrieval Test Interface - Detailed Inspection</h1>

            <div class="query-info">
                <p><strong>Query:</strong> {q}</p>
                <p><strong>Filters:</strong> {filters}</p>
                <p><strong>Number of results requested:</strong> {k}</p>
                <p><strong>Results retrieved:</strong> {len(results)}</p>
            </div>

            <h2>Retrieved Results</h2>
            {"".join([
                f'''
                <div class="result">
                    <div><strong>Rank {r["rank"]} - Similarity: {r["similarity_score"]:.3f}</strong></div>
                    <div class="meta">ID: {r["id"]}</div>
                    <div class="meta">Chapter: {r["chapter"] or 'N/A'}, Section: {r["section"] or 'N/A'}</div>
                    <div class="meta">Confidence: {r["confidence_score"]:.3f}</div>
                    <div class="meta">Timestamp: {r["retrieval_timestamp"]}</div>
                    <div class="meta">Source: <a href="{r["source_url"]}" target="_blank">{r["source_url"]}</a></div>
                    <div class="content"><strong>Content Preview:</strong> {r["content"]}</div>

                    <button class="toggle-btn" onclick="toggleDetail('detail-{r["rank"]}')">Toggle Full Content</button>
                    <div id="detail-{r["rank"]}" class="detailed-view">
                        <h4>Full Content:</h4>
                        <p>{r["full_content"]}</p>
                    </div>
                </div>
                ''' for r in formatted_results
            ])}

            <hr>
            <p><small>Retrieval validation test interface - Detailed inspection mode</small></p>
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error during test interface: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test interface failed: {str(e)}")


@router.get("/debug-result/{result_id}")
async def debug_retrieval_result(result_id: str):
    """
    Debug endpoint for detailed inspection of a specific retrieval result
    """
    try:
        # This would normally retrieve the specific result from storage
        # For now, we'll return a placeholder response indicating the concept
        html_content = f"""
        <html>
        <head>
            <title>Debug Result: {result_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .info-box {{ border: 1px solid #ddd; padding: 10px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Debug Information for Result: {result_id}</h1>

            <div class="info-box">
                <h3>Result Details</h3>
                <p>This would show detailed information about the specific result:</p>
                <ul>
                    <li>Full content text</li>
                    <li>Source document information</li>
                    <li>Metadata and annotations</li>
                    <li>Similarity scores and confidence metrics</li>
                    <li>Vector representation details</li>
                    <li>Relationship to query terms</li>
                </ul>
            </div>

            <div class="info-box">
                <h3>Query Context</h3>
                <p>Original query that produced this result would be shown here</p>
            </div>

            <div class="info-box">
                <h3>Processing Pipeline</h3>
                <p>Information about how this result was retrieved and ranked would be shown here</p>
            </div>

            <a href="/">Back to Test Interface</a>
        </body>
        </html>
        """

        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error during debug result interface: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug result interface failed: {str(e)}")


@router.post("/basic-validate", response_model=ValidationResponse)
async def basic_validate_retrieval(query: str = Query(..., description="Query text to validate")):
    """
    Basic validation endpoint for retrieval quality without metadata filters
    """
    start_time = time.time()

    try:
        # Create a validation request with no filters
        validation_request = ValidationRequest(
            query=query,
            expected_sources=None,
            filters=None,
            top_k=5
        )

        # Perform retrieval
        results = await retrieval_service.retrieve_similar_chunks(validation_request)

        # Calculate metrics
        latency_ms = (time.time() - start_time) * 1000

        # Calculate semantic alignment score
        semantic_alignment_score = await retrieval_service.calculate_semantic_alignment(query, results)

        # Calculate precision metric
        precision_metric = await retrieval_service.calculate_precision(results, None)

        # Determine if validation passed based on success criteria
        validation_passed = (
            precision_metric >= 0.70 and  # At least 70% precision
            latency_ms <= 1500  # Under 1.5 seconds
        )

        response = ValidationResponse(
            query=query,
            retrieved_results=results,
            semantic_alignment_score=semantic_alignment_score,
            precision_metric=precision_metric,
            latency_ms=latency_ms,
            validation_passed=validation_passed
        )

        logger.info(f"Basic retrieval validation completed for query: '{query[:50]}...', "
                   f"latency: {latency_ms:.2f}ms, results: {len(results)}")

        return response

    except Exception as e:
        logger.error(f"Error during basic retrieval validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Basic retrieval validation failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the validation service
    """
    try:
        # Check if the retrieval service is working
        health_status = await retrieval_service.health_check()

        return {
            "status": "healthy" if health_status else "unhealthy",
            "service": "retrieval-validation",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "retrieval-validation",
            "error": str(e),
            "timestamp": time.time()
        }


from fastapi.responses import HTMLResponse