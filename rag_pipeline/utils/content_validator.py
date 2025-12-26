"""
Content validation utilities
Validates extracted content and chunks to ensure quality
"""
import re
from typing import List, Optional
from ..models.book_content import BookContent
from ..models.content_chunk import ContentChunk
from ..utils.logging_config import get_logger
from ..utils.exceptions import ValidationError


class ContentValidator:
    """
    Utility class for validating content and chunks
    """
    def __init__(self):
        self.logger = get_logger()

    def validate_book_content(self, book_content: BookContent) -> bool:
        """
        Validate a BookContent object

        Args:
            book_content: BookContent object to validate

        Returns:
            True if valid, raises ValidationError if invalid
        """
        errors = []

        # Check ID
        if not book_content.id or not isinstance(book_content.id, str) or len(book_content.id.strip()) == 0:
            errors.append("ID must be a non-empty string")

        # Check URL
        if not book_content.url or not isinstance(book_content.url, str) or len(book_content.url.strip()) == 0:
            errors.append("URL must be a non-empty string")

        # Check title
        if not book_content.title or not isinstance(book_content.title, str) or len(book_content.title.strip()) == 0:
            errors.append("Title must be a non-empty string")

        # Check content
        if not book_content.content or not isinstance(book_content.content, str) or len(book_content.content.strip()) == 0:
            errors.append("Content must be a non-empty string")

        # Check content length
        if len(book_content.content.strip()) < 10:
            errors.append("Content is too short (less than 10 characters)")

        if errors:
            error_msg = f"BookContent validation failed: {'; '.join(errors)}"
            self.logger.error(error_msg)
            raise ValidationError(error_msg)

        self.logger.debug(f"BookContent validation passed for ID: {book_content.id}")
        return True

    def validate_content_chunk(self, content_chunk: ContentChunk) -> bool:
        """
        Validate a ContentChunk object

        Args:
            content_chunk: ContentChunk object to validate

        Returns:
            True if valid, raises ValidationError if invalid
        """
        errors = []

        # Check ID
        if not content_chunk.id or not isinstance(content_chunk.id, str) or len(content_chunk.id.strip()) == 0:
            errors.append("ID must be a non-empty string")

        # Check book_content_id
        if not content_chunk.book_content_id or not isinstance(content_chunk.book_content_id, str) or len(content_chunk.book_content_id.strip()) == 0:
            errors.append("book_content_id must be a non-empty string")

        # Check content
        if not content_chunk.content or not isinstance(content_chunk.content, str) or len(content_chunk.content.strip()) == 0:
            errors.append("Content must be a non-empty string")

        # Check chunk_index
        if not isinstance(content_chunk.chunk_index, int) or content_chunk.chunk_index < 0:
            errors.append("chunk_index must be a non-negative integer")

        # Check word_count
        if not isinstance(content_chunk.word_count, int) or content_chunk.word_count < 0:
            errors.append("word_count must be a non-negative integer")

        # Check char_count
        if not isinstance(content_chunk.char_count, int) or content_chunk.char_count < 0:
            errors.append("char_count must be a non-negative integer")

        # Check content length vs char_count
        actual_char_count = len(content_chunk.content)
        if actual_char_count != content_chunk.char_count:
            errors.append(f"Content character count mismatch: expected {content_chunk.char_count}, got {actual_char_count}")

        # Check content vs word_count
        actual_word_count = len(content_chunk.content.split())
        if actual_word_count != content_chunk.word_count:
            errors.append(f"Content word count mismatch: expected {content_chunk.word_count}, got {actual_word_count}")

        if errors:
            error_msg = f"ContentChunk validation failed: {'; '.join(errors)}"
            self.logger.error(error_msg)
            raise ValidationError(error_msg)

        self.logger.debug(f"ContentChunk validation passed for ID: {content_chunk.id}")
        return True

    def validate_content_chunks(self, content_chunks: List[ContentChunk]) -> bool:
        """
        Validate a list of ContentChunk objects

        Args:
            content_chunks: List of ContentChunk objects to validate

        Returns:
            True if all chunks are valid, raises ValidationError if any are invalid
        """
        if not content_chunks:
            raise ValidationError("Content chunks list cannot be empty")

        # Check for duplicate IDs
        chunk_ids = [chunk.id for chunk in content_chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValidationError("Duplicate chunk IDs found")

        # Validate each chunk
        for chunk in content_chunks:
            self.validate_content_chunk(chunk)

        self.logger.info(f"All {len(content_chunks)} content chunks validated successfully")
        return True

    def validate_chunk_semantic_coherence(self, content_chunk: ContentChunk) -> bool:
        """
        Validate that a chunk maintains semantic coherence

        Args:
            content_chunk: ContentChunk object to validate

        Returns:
            True if semantically coherent, False otherwise
        """
        content = content_chunk.content.strip()

        # Check if the content starts with a sentence boundary
        starts_with_sentence_boundary = content[0].isupper() if content else False

        # Check if the content ends with a sentence boundary
        ends_with_sentence_boundary = content[-1] in '.!?' if content else False

        # Check for incomplete sentences (if content is short, it might be a heading)
        if len(content) > 50:  # Only check for longer chunks
            incomplete_sentence = not ends_with_sentence_boundary
        else:
            incomplete_sentence = False

        # Additional checks for semantic boundaries
        # For example, check if chunk ends mid-sentence or mid-paragraph
        has_proper_ending = (
            content.endswith('.') or
            content.endswith('!') or
            content.endswith('?') or
            content.endswith('</p>') or
            content.endswith('</div>') or
            content.endswith('\n\n') or
            content.endswith('\n')  # If it ends with newline, it might be intentional
        )

        # If the chunk is too short relative to its expected content, it might be broken
        too_short = len(content) < 10 and not content.replace(' ', '').replace('\n', '').replace('\t', '').isnumeric()

        # Determine if chunk is semantically coherent
        is_coherent = (
            not incomplete_sentence or  # Allow incomplete sentences if it's a special case
            has_proper_ending or  # Or has proper ending markers
            too_short  # Or is intentionally short (like a heading)
        )

        if not is_coherent:
            self.logger.warning(f"Chunk {content_chunk.id} may have semantic coherence issues")

        return is_coherent

    def validate_text_quality(self, text: str, min_length: int = 10) -> bool:
        """
        Validate the quality of a text string

        Args:
            text: Text to validate
            min_length: Minimum length for valid text

        Returns:
            True if text quality is acceptable, False otherwise
        """
        if not text or len(text.strip()) < min_length:
            return False

        # Check for excessive special characters or non-alphanumeric content
        clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        if len(clean_text.strip()) == 0:
            return False

        # Check if text is mostly whitespace or special characters
        alphanumeric_ratio = len(clean_text) / len(text)
        if alphanumeric_ratio < 0.3:  # Less than 30% alphanumeric
            return False

        return True