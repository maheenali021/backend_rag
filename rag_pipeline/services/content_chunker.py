"""
Content chunking service for RAG Pipeline
Chunks extracted content using semantic boundaries
"""
import re
from typing import List, Optional
from ..models.book_content import BookContent
from ..models.content_chunk import ContentChunk
from ..utils.logging_config import get_logger
from ..utils.exceptions import ContentChunkingError
from ..utils.helpers import generate_id, sanitize_text
from ..config import Config


class ContentChunker:
    """
    Service class for chunking content into semantically meaningful pieces
    """
    def __init__(self):
        self.logger = get_logger()
        self.config = Config

    def chunk_content(self, book_content: BookContent, chunk_size: Optional[int] = None,
                      overlap_size: Optional[int] = None) -> List[ContentChunk]:
        """
        Chunk the content of a BookContent object

        Args:
            book_content: BookContent object to chunk
            chunk_size: Size of each chunk (optional, defaults to config)
            overlap_size: Size of overlap between chunks (optional, defaults to config)

        Returns:
            List of ContentChunk objects
        """
        chunk_size = chunk_size or self.config.CHUNK_SIZE
        overlap_size = overlap_size or self.config.CHUNK_OVERLAP

        try:
            content = book_content.content
            if not content:
                self.logger.warning(f"No content to chunk for {book_content.id}")
                return []

            # Split content into chunks based on semantic boundaries
            chunks = self._semantic_chunking(content, chunk_size, overlap_size, book_content.id)

            # Extract chapter/section information from URL or title
            source_url = book_content.url
            chapter = self._extract_chapter_from_url(book_content.url, book_content.title)
            section = self._extract_section_from_title(book_content.title)

            # Create ContentChunk objects
            content_chunks = []
            for i, (chunk_text, boundary_type) in enumerate(chunks):
                chunk = ContentChunk(
                    id=generate_id(f"{book_content.id}_{i}_{chunk_text[:50]}", prefix="cc_"),
                    book_content_id=book_content.id,
                    content=chunk_text,
                    chunk_index=i,
                    semantic_boundary=boundary_type,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    source_url=source_url,
                    chapter=chapter,
                    section=section
                )
                content_chunks.append(chunk)

            self.logger.info(f"Chunked content {book_content.id} into {len(content_chunks)} chunks")
            return content_chunks

        except Exception as e:
            self.logger.error(f"Error chunking content {book_content.id}: {str(e)}")
            raise ContentChunkingError(f"Error chunking content {book_content.id}: {str(e)}")

    def _semantic_chunking(self, content: str, chunk_size: int, overlap_size: int,
                          book_content_id: str) -> List[tuple]:
        """
        Perform semantic chunking on content

        Args:
            content: Content to chunk
            chunk_size: Target size of each chunk
            overlap_size: Size of overlap between chunks
            book_content_id: ID of the parent book content

        Returns:
            List of tuples containing (chunk_text, boundary_type)
        """
        # First, try to split by semantic boundaries
        semantic_splits = self._split_by_semantic_boundaries(content)

        chunks = []
        current_chunk = ""
        current_boundary = "unknown"

        for part, boundary_type in semantic_splits:
            # If adding this part would exceed the chunk size, start a new chunk
            if len(current_chunk) + len(part) > chunk_size and current_chunk:
                # Add the current chunk to the list
                if current_chunk.strip():
                    chunks.append((current_chunk.strip(), current_boundary))

                # Start a new chunk, possibly with overlap
                if overlap_size > 0 and chunks:
                    # Add overlap from the previous chunk
                    last_chunk = chunks[-1][0]
                    overlap = last_chunk[-overlap_size:]
                    current_chunk = overlap + " " + part
                else:
                    current_chunk = part
                current_boundary = boundary_type
            else:
                # Add the part to the current chunk
                if current_chunk:
                    current_chunk += " " + part
                else:
                    current_chunk = part
                current_boundary = boundary_type

        # Add the final chunk if it has content
        if current_chunk.strip():
            chunks.append((current_chunk.strip(), current_boundary))

        # If any chunks are still too large, fall back to character-based chunking
        final_chunks = []
        for chunk_text, boundary_type in chunks:
            if len(chunk_text) > chunk_size:
                # Split the large chunk further using character-based chunking
                sub_chunks = self._character_based_chunking(chunk_text, chunk_size, overlap_size)
                for sub_chunk in sub_chunks:
                    final_chunks.append((sub_chunk, f"split_from_{boundary_type}"))
            else:
                final_chunks.append((chunk_text, boundary_type))

        return final_chunks

    def _split_by_semantic_boundaries(self, content: str) -> List[tuple]:
        """
        Split content by semantic boundaries like headings, paragraphs, etc.

        Args:
            content: Content to split

        Returns:
            List of tuples containing (content_part, boundary_type)
        """
        parts = []

        # Define semantic boundary patterns
        boundary_patterns = [
            (r'(?:^|\n)#{1,6}\s+.*?(?=\n|$)', 'heading'),  # Markdown headings
            (r'(?:^|\n)={1,6}\s+.*?(?=\n|$)', 'title'),   # Title underlines
            (r'(?:^|\n)={3,}\n.*?\n={3,}(?=\n|$)', 'section'),  # Section dividers
            (r'(?:^|\n)\d+\.\s+.*?(?=\n|$)', 'numbered_list_item'),  # Numbered list items
            (r'(?:^|\n)[a-zA-Z]\.\s+.*?(?=\n|$)', 'lettered_list_item'),  # Lettered list items
            (r'(?:^|\n)\d+\)\s+.*?(?=\n|$)', 'numbered_item'),  # Numbered items with parentheses
            (r'(?:^|\n)\s*\n\s*\n', 'paragraph_boundary'),  # Paragraph breaks
        ]

        # Try each boundary pattern
        current_pos = 0
        content_length = len(content)

        # First, let's identify all the boundary positions
        boundaries = []
        for pattern, boundary_type in boundary_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
                boundaries.append((match.start(), match.end(), boundary_type))

        # Sort boundaries by position
        boundaries.sort(key=lambda x: x[0])

        # Process content based on boundaries
        last_end = 0
        for start, end, boundary_type in boundaries:
            # Add content before this boundary
            if start > last_end:
                content_before = content[last_end:start].strip()
                if content_before:
                    parts.append((content_before, 'content'))

            # Add the boundary content
            boundary_content = content[start:end].strip()
            if boundary_content:
                parts.append((boundary_content, boundary_type))

            last_end = end

        # Add remaining content after the last boundary
        if last_end < content_length:
            remaining_content = content[last_end:].strip()
            if remaining_content:
                parts.append((remaining_content, 'content'))

        # If no semantic boundaries were found, fall back to paragraph-based splitting
        if not parts:
            # Split by paragraphs (double newlines)
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    parts.append((para.strip(), 'paragraph'))

        return parts

    def _character_based_chunking(self, content: str, chunk_size: int, overlap_size: int) -> List[str]:
        """
        Fallback method to chunk content by character count

        Args:
            content: Content to chunk
            chunk_size: Target size of each chunk
            overlap_size: Size of overlap between chunks

        Returns:
            List of chunk strings
        """
        if len(content) <= chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size

            # If we're at the end, take the remaining content
            if end >= len(content):
                chunks.append(content[start:])
                break

            # Try to break at sentence or word boundary
            chunk = content[start:end]

            # Find the last sentence end within the chunk
            sentence_end = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))

            if sentence_end != -1 and sentence_end > len(chunk) // 2:  # If sentence end is in the latter half
                end = start + sentence_end + 1
            else:
                # Find the last word boundary
                word_end = chunk.rfind(' ')
                if word_end != -1 and word_end > len(chunk) // 2:  # If word end is in the latter half
                    end = start + word_end

            chunk = content[start:end]
            chunks.append(chunk)

            # Move start position, considering overlap
            start = end - overlap_size if overlap_size > 0 else end

            # Ensure we make progress to avoid infinite loops
            if start <= end - chunk_size:  # If we're not making enough progress
                start = end

        return chunks

    def chunk_multiple_contents(self, book_contents: List[BookContent]) -> List[ContentChunk]:
        """
        Chunk multiple BookContent objects

        Args:
            book_contents: List of BookContent objects to chunk

        Returns:
            List of ContentChunk objects
        """
        all_chunks = []
        for book_content in book_contents:
            try:
                chunks = self.chunk_content(book_content)
                all_chunks.extend(chunks)
            except ContentChunkingError as e:
                self.logger.error(f"Failed to chunk content {book_content.id}: {str(e)}")
                # Continue with other contents even if one fails

        self.logger.info(f"Chunked {len(book_contents)} book contents into {len(all_chunks)} total chunks")
        return all_chunks

    def validate_chunks(self, content_chunks: List[ContentChunk], max_chunk_size: Optional[int] = None) -> bool:
        """
        Validate that chunks meet size requirements

        Args:
            content_chunks: List of ContentChunk objects to validate
            max_chunk_size: Maximum allowed chunk size (optional, defaults to config)

        Returns:
            True if all chunks are valid, False otherwise
        """
        max_chunk_size = max_chunk_size or self.config.CHUNK_SIZE * 1.5  # Allow some flexibility

        for chunk in content_chunks:
            if len(chunk.content) > max_chunk_size:
                self.logger.warning(f"Chunk {chunk.id} exceeds maximum size: {len(chunk.content)} > {max_chunk_size}")
                return False

            if not chunk.content.strip():
                self.logger.warning(f"Chunk {chunk.id} has no content")
                return False

        return True

    def _extract_chapter_from_url(self, url: str, title: str) -> Optional[str]:
        """
        Extract chapter information from URL or title

        Args:
            url: The URL of the content
            title: The title of the content

        Returns:
            Chapter name or None if not found
        """
        # Try to extract chapter from URL path
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')

        # Look for common chapter-like patterns in URL
        for part in path_parts:
            # Look for patterns like 'module1', 'chapter2', 'docs/module1', etc.
            if any(keyword in part.lower() for keyword in ['module', 'chapter', 'lesson', 'part']):
                return part

        # If no chapter found in URL, try to extract from title
        lower_title = title.lower()
        if 'module' in lower_title or 'chapter' in lower_title:
            # Try to extract chapter/module info from title
            import re
            chapter_match = re.search(r'(?:module|chapter|lesson)\s*(\d+(?:\.\d+)?)', title, re.IGNORECASE)
            if chapter_match:
                return f"Module {chapter_match.group(1)}"

        return None

    def _extract_section_from_title(self, title: str) -> Optional[str]:
        """
        Extract section information from title

        Args:
            title: The title of the content

        Returns:
            Section name or None if not found
        """
        # Remove common chapter/module prefixes to get the actual section title
        import re
        # Remove chapter/module prefixes
        clean_title = re.sub(r'^(?:module\s*\d+[\.\-\s]*)|(?:chapter\s*\d+[\.\-\s]*)|(?:lesson\s*\d+[\.\-\s]*)', '', title, flags=re.IGNORECASE)
        clean_title = clean_title.strip()

        # Return the cleaned title as section if it's meaningful
        if clean_title and len(clean_title) > 2:
            return clean_title

        return None

    def calculate_chunk_metrics(self, content_chunks: List[ContentChunk]) -> dict:
        """
        Calculate metrics for the chunks

        Args:
            content_chunks: List of ContentChunk objects

        Returns:
            Dictionary with chunk metrics
        """
        if not content_chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "total_chars": 0,
                "total_words": 0
            }

        total_chars = sum(len(chunk.content) for chunk in content_chunks)
        total_words = sum(chunk.word_count for chunk in content_chunks)
        avg_chunk_size = total_chars / len(content_chunks)

        return {
            "total_chunks": len(content_chunks),
            "avg_chunk_size": avg_chunk_size,
            "total_chars": total_chars,
            "total_words": total_words,
            "max_chunk_size": max(len(chunk.content) for chunk in content_chunks),
            "min_chunk_size": min(len(chunk.content) for chunk in content_chunks)
        }