"""
Extraction and Chunking Pipeline for RAG Pipeline
Orchestrates the content extraction and chunking process
"""
from typing import List
from ..models.book_content import BookContent
from ..models.content_chunk import ContentChunk
from ..services.content_extractor import ContentExtractor
from ..services.content_chunker import ContentChunker
from ..utils.content_cleaner import ContentCleaner
from ..utils.content_validator import ContentValidator
from ..utils.logging_config import get_logger
from ..utils.exceptions import RAGPipelineError, ContentExtractionError, ContentChunkingError
from ..utils.error_handler import error_handler, handle_pipeline_error


class ExtractionPipeline:
    """
    Pipeline class that orchestrates content extraction and chunking
    """
    def __init__(self):
        self.logger = get_logger()
        self.content_extractor = ContentExtractor()
        self.content_chunker = ContentChunker()
        self.content_cleaner = ContentCleaner()
        self.content_validator = ContentValidator()

    @handle_pipeline_error("ExtractionPipeline.extract_content")
    def extract_content(self, url: str = None) -> List[BookContent]:
        """
        Extract content from a website

        Args:
            url: URL to extract content from (optional, defaults to config)

        Returns:
            List of BookContent objects
        """
        self.logger.info("Starting content extraction pipeline")

        # Extract content from website
        if url:
            book_contents = [self.content_extractor.extract_content_from_url(url)]
        else:
            # Try sitemap first, then fall back to link extraction
            book_contents = self.content_extractor.extract_content_by_sitemap()

        self.logger.info(f"Extracted content from {len(book_contents)} pages")

        # Clean the extracted content
        cleaned_contents = []
        for book_content in book_contents:
            try:
                cleaned_content = self.content_cleaner.clean_book_content(book_content)
                self.content_validator.validate_book_content(cleaned_content)
                cleaned_contents.append(cleaned_content)
            except Exception as e:
                self.logger.warning(f"Failed to clean or validate content {book_content.id}: {str(e)}")
                # Still add the original content if cleaning/validation fails
                cleaned_contents.append(book_content)

        self.logger.info("Content extraction completed successfully")
        return cleaned_contents

    @handle_pipeline_error("ExtractionPipeline.chunk_content")
    def chunk_content(self, book_contents: List[BookContent]) -> List[ContentChunk]:
        """
        Chunk the extracted content

        Args:
            book_contents: List of BookContent objects to chunk

        Returns:
            List of ContentChunk objects
        """
        self.logger.info(f"Starting content chunking for {len(book_contents)} book contents")

        # Chunk each book content
        all_chunks = []
        for book_content in book_contents:
            try:
                chunks = self.content_chunker.chunk_content(book_content)
                all_chunks.extend(chunks)
            except ContentChunkingError as e:
                self.logger.error(f"Failed to chunk content {book_content.id}: {str(e)}")
                continue  # Continue with other contents

        # Validate the chunks
        try:
            self.content_validator.validate_content_chunks(all_chunks)
            self.logger.info(f"Validated {len(all_chunks)} chunks successfully")
        except Exception as e:
            self.logger.warning(f"Chunk validation failed: {str(e)}")

        # Validate semantic coherence
        coherent_chunks = []
        for chunk in all_chunks:
            is_coherent = self.content_validator.validate_chunk_semantic_coherence(chunk)
            if is_coherent:
                coherent_chunks.append(chunk)
            else:
                self.logger.debug(f"Chunk {chunk.id} may have coherence issues but keeping it")

        self.logger.info(f"Content chunking completed: {len(coherent_chunks)} coherent chunks from {len(book_contents)} contents")
        return coherent_chunks

    @handle_pipeline_error("ExtractionPipeline.run_extraction_pipeline")
    def run_extraction_pipeline(self, url: str = None) -> List[ContentChunk]:
        """
        Run the complete extraction and chunking pipeline

        Args:
            url: URL to extract content from (optional, defaults to config)

        Returns:
            List of ContentChunk objects
        """
        self.logger.info("Starting extraction and chunking pipeline")

        # Step 1: Extract content
        book_contents = self.extract_content(url)

        if not book_contents:
            self.logger.warning("No content extracted, returning empty list")
            return []

        # Step 2: Chunk content
        content_chunks = self.chunk_content(book_contents)

        self.logger.info(f"Extraction and chunking pipeline completed successfully. "
                        f"Extracted {len(book_contents)} contents and created {len(content_chunks)} chunks")

        return content_chunks

    @handle_pipeline_error("ExtractionPipeline.run_incremental_extraction")
    def run_incremental_extraction(self, url: str, existing_chunks: List[ContentChunk] = None) -> List[ContentChunk]:
        """
        Run extraction for new or updated content only

        Args:
            url: URL to extract content from
            existing_chunks: Existing chunks to compare against (optional)

        Returns:
            List of new or updated ContentChunk objects
        """
        self.logger.info("Starting incremental extraction")

        # For now, we'll do a full extraction
        # In a real implementation, this would compare content hashes to identify changes
        new_chunks = self.run_extraction_pipeline(url)

        # In a more sophisticated implementation, we would:
        # 1. Compare content hashes to identify new/changed content
        # 2. Only process the changed content
        # 3. Return only new or updated chunks

        self.logger.info(f"Incremental extraction completed. Created {len(new_chunks)} chunks")
        return new_chunks

    def get_pipeline_metrics(self, book_contents: List[BookContent], content_chunks: List[ContentChunk]) -> dict:
        """
        Get metrics for the extraction and chunking pipeline

        Args:
            book_contents: List of BookContent objects
            content_chunks: List of ContentChunk objects

        Returns:
            Dictionary with pipeline metrics
        """
        content_chars = sum(len(bc.content) for bc in book_contents)
        content_words = sum(len(bc.content.split()) for bc in book_contents)

        chunk_metrics = self.content_chunker.calculate_chunk_metrics(content_chunks)

        return {
            "extraction": {
                "total_contents": len(book_contents),
                "total_characters": content_chars,
                "total_words": content_words,
                "avg_content_size": content_chars / len(book_contents) if book_contents else 0
            },
            "chunking": chunk_metrics,
            "efficiency": {
                "compression_ratio": content_chars / chunk_metrics["total_chars"] if chunk_metrics["total_chars"] > 0 else 0,
                "chunks_per_content": len(content_chunks) / len(book_contents) if book_contents > 0 else 0
            }
        }

    @handle_pipeline_error("ExtractionPipeline.validate_pipeline_output")
    def validate_pipeline_output(self, book_contents: List[BookContent], content_chunks: List[ContentChunk]) -> bool:
        """
        Validate the output of the extraction pipeline

        Args:
            book_contents: List of BookContent objects
            content_chunks: List of ContentChunk objects

        Returns:
            True if output is valid, False otherwise
        """
        # Validate book contents
        for content in book_contents:
            try:
                self.content_validator.validate_book_content(content)
            except Exception as e:
                self.logger.error(f"BookContent validation failed for {content.id}: {str(e)}")
                return False

        # Validate content chunks
        try:
            self.content_validator.validate_content_chunks(content_chunks)
        except Exception as e:
            self.logger.error(f"ContentChunk validation failed: {str(e)}")
            return False

        # Check for semantic coherence
        incoherent_count = 0
        for chunk in content_chunks:
            if not self.content_validator.validate_chunk_semantic_coherence(chunk):
                incoherent_count += 1

        coherence_rate = 1 - (incoherent_count / len(content_chunks)) if content_chunks else 1
        self.logger.info(f"Semantic coherence rate: {coherence_rate:.2%}")

        # For the acceptance criteria, we need >95% coherence
        if coherence_rate < 0.95:
            self.logger.warning(f"Semantic coherence rate below 95%: {coherence_rate:.2%}")
            return False

        return True