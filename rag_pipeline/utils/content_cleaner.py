"""
Content cleaning and normalization utilities
Cleans and normalizes extracted content for better processing
"""
import re
from typing import List, Tuple
from ..models.book_content import BookContent
from ..models.content_chunk import ContentChunk
from ..utils.logging_config import get_logger
from ..utils.exceptions import ContentChunkingError
from ..utils.helpers import sanitize_text


class ContentCleaner:
    """
    Utility class for cleaning and normalizing content
    """
    def __init__(self):
        self.logger = get_logger()

    def clean_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text

        Args:
            text: Text with potential HTML tags

        Returns:
            Text with HTML tags removed
        """
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', ' ', text)
        # Replace multiple spaces with single space
        clean_text = ' '.join(clean_text.split())
        return clean_text.strip()

    def clean_code_blocks(self, text: str) -> str:
        """
        Clean code blocks from text while preserving important content

        Args:
            text: Text with potential code blocks

        Returns:
            Text with cleaned code blocks
        """
        # Remove common code block markers but preserve the content
        # Handle Markdown-style code blocks
        text = re.sub(r'```.*?\n(.*?)```', r'\1', text, flags=re.DOTALL)
        # Handle single backticks
        text = re.sub(r'`(.*?)`', r'\1', text)

        return text

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text

        Args:
            text: Text to normalize

        Returns:
            Text with normalized whitespace
        """
        # Replace various whitespace characters with standard spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()

    def remove_boilerplate_content(self, text: str) -> str:
        """
        Remove common boilerplate content like navigation, headers, footers

        Args:
            text: Text to clean

        Returns:
            Text with boilerplate content removed
        """
        # Common boilerplate patterns to remove
        boilerplate_patterns = [
            r'navigation',
            r'menu',
            r'sidebar',
            r'footer',
            r'header',
            r'copyright',
            r'©.*?\d{4}',
            r'all rights reserved',
            r'privacy policy',
            r'terms of service',
            r'cookie policy',
            r'sitemap',
            r'home',
            r'contact us',
            r'about us',
        ]

        # Remove boilerplate patterns
        for pattern in boilerplate_patterns:
            # Case-insensitive removal
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return self.normalize_whitespace(text)

    def clean_and_normalize(self, text: str) -> str:
        """
        Apply all cleaning and normalization operations

        Args:
            text: Text to clean and normalize

        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""

        # Apply cleaning operations in sequence
        text = self.clean_html_tags(text)
        text = self.clean_code_blocks(text)
        text = self.remove_boilerplate_content(text)
        text = self.normalize_whitespace(text)

        # Additional sanitization
        text = sanitize_text(text)

        return text

    def clean_book_content(self, book_content: BookContent) -> BookContent:
        """
        Clean and normalize a BookContent object

        Args:
            book_content: BookContent object to clean

        Returns:
            Cleaned BookContent object
        """
        try:
            cleaned_content = self.clean_and_normalize(book_content.content)

            # Update the content with cleaned version
            cleaned_book_content = BookContent(
                id=book_content.id,
                url=book_content.url,
                title=book_content.title,
                content=cleaned_content,
                metadata=book_content.metadata,
                created_at=book_content.created_at,
                updated_at=book_content.updated_at
            )

            self.logger.info(f"Cleaned content for {book_content.id}, original length: {len(book_content.content)}, "
                           f"cleaned length: {len(cleaned_content)}")

            return cleaned_book_content

        except Exception as e:
            self.logger.error(f"Error cleaning BookContent {book_content.id}: {str(e)}")
            raise ContentChunkingError(f"Error cleaning BookContent {book_content.id}: {str(e)}")

    def clean_content_chunks(self, content_chunks: List[ContentChunk]) -> List[ContentChunk]:
        """
        Clean and normalize a list of ContentChunk objects

        Args:
            content_chunks: List of ContentChunk objects to clean

        Returns:
            List of cleaned ContentChunk objects
        """
        cleaned_chunks = []
        for chunk in content_chunks:
            try:
                cleaned_content = self.clean_and_normalize(chunk.content)

                cleaned_chunk = ContentChunk(
                    id=chunk.id,
                    book_content_id=chunk.book_content_id,
                    content=cleaned_content,
                    chunk_index=chunk.chunk_index,
                    semantic_boundary=chunk.semantic_boundary,
                    word_count=len(cleaned_content.split()),
                    char_count=len(cleaned_content),
                    created_at=chunk.created_at,
                    updated_at=chunk.updated_at
                )

                cleaned_chunks.append(cleaned_chunk)

            except Exception as e:
                self.logger.error(f"Error cleaning ContentChunk {chunk.id}: {str(e)}")
                raise ContentChunkingError(f"Error cleaning ContentChunk {chunk.id}: {str(e)}")

        self.logger.info(f"Cleaned {len(cleaned_chunks)} content chunks")
        return cleaned_chunks

    def extract_meaningful_content(self, text: str) -> str:
        """
        Extract only meaningful content, removing excessive punctuation and special characters

        Args:
            text: Text to process

        Returns:
            Text with meaningful content preserved
        """
        # Remove excessive special characters but preserve important punctuation
        # Keep letters, numbers, common punctuation, and some special characters
        meaningful_text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\'\"\/]', ' ', text)

        # Remove excessive repeated characters (e.g., multiple hyphens, underscores)
        meaningful_text = re.sub(r'([-_]){2,}', r'\1', meaningful_text)

        return self.normalize_whitespace(meaningful_text)

    def preserve_content_structure(self, text: str) -> str:
        """
        Preserve important content structure like headings and paragraphs

        Args:
            text: Text to process

        Returns:
            Text with structure preserved
        """
        # Normalize different heading formats to a standard format
        # This could involve identifying patterns that represent headings
        # and ensuring they are properly separated

        # Common heading patterns in documentation
        patterns = [
            (r'^\s*#{1,6}\s+(.*?)$', r'\nHEADING: \1\n'),  # Markdown headings
            (r'^\s*\d+[\.\)]\s+(.*?)$', r'\nSUBHEADING: \1\n'),  # Numbered lists that might be sections
        ]

        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)

        return text