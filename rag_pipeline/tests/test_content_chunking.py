"""
Unit tests for content chunking module
"""
import unittest
from rag_pipeline.services.content_chunker import ContentChunker
from rag_pipeline.models.book_content import BookContent
from rag_pipeline.models.content_chunk import ContentChunk


class TestContentChunker(unittest.TestCase):
    def setUp(self):
        self.content_chunker = ContentChunker()

    def test_chunk_content_basic(self):
        """Test basic content chunking"""
        book_content = BookContent(
            id="test_id",
            url="http://example.com",
            title="Test Title",
            content="This is the first sentence. This is the second sentence. " * 50  # Create longer content
        )

        chunks = self.content_chunker.chunk_content(book_content, chunk_size=100, overlap_size=20)

        # Should have multiple chunks since content is longer than chunk size
        self.assertGreater(len(chunks), 1)

        # Each chunk should be a ContentChunk object
        for chunk in chunks:
            self.assertIsInstance(chunk, ContentChunk)
            self.assertLessEqual(len(chunk.content), 100)  # Should respect chunk size

    def test_chunk_content_small(self):
        """Test chunking content that's smaller than chunk size"""
        book_content = BookContent(
            id="test_id",
            url="http://example.com",
            title="Test Title",
            content="This is short content."
        )

        chunks = self.content_chunker.chunk_content(book_content, chunk_size=1000)

        # Should have only one chunk
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "This is short content.")

    def test_chunk_content_empty(self):
        """Test chunking empty content"""
        book_content = BookContent(
            id="test_id",
            url="http://example.com",
            title="Test Title",
            content=""
        )

        chunks = self.content_chunker.chunk_content(book_content)
        self.assertEqual(len(chunks), 0)

    def test_validate_chunks_within_size(self):
        """Test chunk validation for size compliance"""
        test_chunks = [
            ContentChunk(
                id=f"test_chunk_{i}",
                book_content_id="test_content",
                content="A" * 500,  # 500 characters
                chunk_index=i,
                word_count=71,  # ~71 words for 500 chars
                char_count=500
            )
            for i in range(3)
        ]

        is_valid = self.content_chunker.validate_chunks(test_chunks, max_chunk_size=600)
        self.assertTrue(is_valid)

    def test_validate_chunks_exceed_size(self):
        """Test chunk validation for size violation"""
        test_chunks = [
            ContentChunk(
                id="test_chunk_1",
                book_content_id="test_content",
                content="A" * 1000,  # 1000 characters
                chunk_index=0,
                word_count=142,  # ~142 words for 1000 chars
                char_count=1000
            )
        ]

        is_valid = self.content_chunker.validate_chunks(test_chunks, max_chunk_size=600)
        self.assertFalse(is_valid)

    def test_calculate_chunk_metrics(self):
        """Test chunk metrics calculation"""
        test_chunks = [
            ContentChunk(
                id="test_chunk_1",
                book_content_id="test_content",
                content="First chunk content here",
                chunk_index=0,
                word_count=4,
                char_count=23
            ),
            ContentChunk(
                id="test_chunk_2",
                book_content_id="test_content",
                content="Second chunk content is longer than the first",
                chunk_index=1,
                word_count=8,
                char_count=46
            )
        ]

        metrics = self.content_chunker.calculate_chunk_metrics(test_chunks)

        self.assertEqual(metrics["total_chunks"], 2)
        self.assertEqual(metrics["total_chars"], 69)  # 23 + 46
        self.assertEqual(metrics["total_words"], 12)  # 4 + 8
        self.assertEqual(metrics["max_chunk_size"], 46)
        self.assertEqual(metrics["min_chunk_size"], 23)


if __name__ == '__main__':
    unittest.main()