"""
Retrieval tool for the RAG Agent system
Implements tool calling to invoke the retrieval pipeline
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from ..config import Config
from rag_pipeline.validation.services.retrieval_service import RetrievalService


class RetrievalTool:
    """
    Tool for invoking the retrieval pipeline to get relevant book content
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config
        # Initialize the direct retrieval service
        try:
            # Import the service function and use it
            from rag_pipeline.validation.services.retrieval_service import get_retrieval_service
            self.retrieval_service = get_retrieval_service()
            self.logger.info("Direct retrieval service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize retrieval service: {str(e)}")
            self.retrieval_service = None

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant content using the retrieval pipeline

        Args:
            query: Search query text
            filters: Optional metadata filters (e.g., chapter, section)
            top_k: Number of results to return

        Returns:
            List of retrieved content chunks with metadata
        """
        try:
            if not self.retrieval_service:
                self.logger.error("Retrieval service not initialized")
                return []

            # Create a validation request for the retrieval
            from rag_pipeline.validation.models.validation_models import ValidationRequest
            validation_request = ValidationRequest(
                query=query,
                expected_sources=None,  # Not needed for basic retrieval
                filters=filters,
                top_k=top_k
            )

            # Run the retrieval asynchronously
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If there's no event loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Retrieve similar chunks using the direct service
            retrieval_results = loop.run_until_complete(
                self.retrieval_service.retrieve_similar_chunks(validation_request)
            )

            # Convert the RetrievalResult objects to the expected dictionary format
            processed_results = []
            for result in retrieval_results:
                processed_item = {
                    "id": result.id,
                    "content": result.content,
                    "source_url": result.source_url,
                    "chapter": result.chapter,
                    "section": result.section,
                    "similarity_score": result.similarity_score,
                    "confidence_score": result.confidence_score,
                    "retrieval_timestamp": result.retrieval_timestamp.isoformat() if hasattr(result.retrieval_timestamp, 'isoformat') else str(result.retrieval_timestamp),
                    "metadata": {}
                }
                processed_results.append(processed_item)

            self.logger.info(f"Retrieved {len(processed_results)} results for query: {query[:50]}...")
            return processed_results

        except Exception as e:
            self.logger.error(f"Error in direct retrieval: {str(e)}")
            return []

    async def async_search(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Async version of search for use in async contexts
        """
        try:
            if not self.retrieval_service:
                self.logger.error("Retrieval service not initialized")
                return []

            # Create a validation request for the retrieval
            from rag_pipeline.validation.models.validation_models import ValidationRequest
            validation_request = ValidationRequest(
                query=query,
                expected_sources=None,  # Not needed for basic retrieval
                filters=filters,
                top_k=top_k
            )

            # Retrieve similar chunks using the direct service
            retrieval_results = await self.retrieval_service.retrieve_similar_chunks(validation_request)

            # Convert the RetrievalResult objects to the expected dictionary format
            processed_results = []
            for result in retrieval_results:
                processed_item = {
                    "id": result.id,
                    "content": result.content,
                    "source_url": result.source_url,
                    "chapter": result.chapter,
                    "section": result.section,
                    "similarity_score": result.similarity_score,
                    "confidence_score": result.confidence_score,
                    "retrieval_timestamp": result.retrieval_timestamp.isoformat() if hasattr(result.retrieval_timestamp, 'isoformat') else str(result.retrieval_timestamp),
                    "metadata": {}
                }
                processed_results.append(processed_item)

            self.logger.info(f"Retrieved {len(processed_results)} results for query: {query[:50]}...")
            return processed_results

        except Exception as e:
            self.logger.error(f"Error in direct retrieval: {str(e)}")
            return []

    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant content using the retrieval pipeline
        Now using a thread-safe approach for async calls
        """
        import concurrent.futures
        import threading

        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're already in an event loop, need to use a different approach
            # Run the async search in a separate thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_async_search_in_thread, query, filters, top_k)
                return future.result()
        except RuntimeError:
            # No event loop running, we can use asyncio.run directly
            return asyncio.run(self.async_search(query, filters, top_k))

    def _run_async_search_in_thread(self, query: str, filters: Optional[Dict[str, Any]], top_k: int):
        """
        Helper method to run async search in a separate thread
        """
        # Create a new event loop for the thread
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(self.async_search(query, filters, top_k))
        finally:
            new_loop.close()

    def validate_retrieval_connection(self) -> bool:
        """
        Validate that the retrieval service is accessible
        """
        try:
            if not self.retrieval_service:
                return False

            # Run the health check asynchronously
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # If we're already in an event loop, we need to handle it differently
                # For now, we'll just return True if the service exists as a basic check
                return True
            except RuntimeError:
                # No event loop running, safe to create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.retrieval_service.health_check())
                finally:
                    loop.close()
        except Exception as e:
            print(f"Error in validate_retrieval_connection: {e}")
            return False

    def search_by_chapter(self, query: str, chapter: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search specifically within a chapter

        Args:
            query: Search query text
            chapter: Chapter to search within
            top_k: Number of results to return

        Returns:
            List of retrieved content chunks from the specified chapter
        """
        filters = {"chapter": chapter}
        return self.search(query, filters, top_k)

    def search_with_filters(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search with custom filters

        Args:
            query: Search query text
            filters: Dictionary of metadata filters
            top_k: Number of results to return

        Returns:
            List of retrieved content chunks matching the filters
        """
        return self.search(query, filters, top_k)

    def get_content_by_source_url(self, source_url: str) -> List[Dict[str, Any]]:
        """
        Retrieve content by source URL

        Args:
            source_url: URL of the source document

        Returns:
            List of content chunks from the specified source
        """
        filters = {"source_url": source_url}
        return self.search("all content", filters, top_k=20)  # Get all content from source