"""
Unit tests for content extraction module
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from rag_pipeline.services.content_extractor import ContentExtractor
from rag_pipeline.models.book_content import BookContent


class TestContentExtractor(unittest.TestCase):
    def setUp(self):
        self.content_extractor = ContentExtractor()

    @patch('rag_pipeline.services.content_extractor.requests.Session.get')
    @patch('rag_pipeline.services.content_extractor.BeautifulSoup')
    def test_extract_content_from_url_success(self, mock_bs, mock_get):
        """Test successful content extraction from a URL"""
        # Mock the response
        mock_response = Mock()
        mock_response.content = b"<html><head><title>Test Title</title></head><body><p>Test content</p></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Mock BeautifulSoup
        mock_soup = Mock()
        mock_title_tag = Mock()
        mock_title_tag.get_text.return_value = "Test Title"
        mock_soup.find.return_value = mock_title_tag
        mock_soup.__str__ = lambda x: "<html>mocked html</html>"
        mock_bs.return_value = mock_soup

        # Mock body content
        mock_body = Mock()
        mock_body.get_text.return_value = "Test content"
        mock_soup.find.return_value = mock_title_tag  # title
        mock_soup.find.side_effect = lambda tag: mock_body if tag == 'body' else mock_title_tag

        # Run the extraction
        result = self.content_extractor.extract_content_from_url("http://example.com")

        # Assertions
        self.assertIsInstance(result, BookContent)
        self.assertEqual(result.title, "Test Title")
        self.assertIn("Test content", result.content)

    @patch('rag_pipeline.services.content_extractor.requests.Session.get')
    def test_extract_content_from_url_network_error(self, mock_get):
        """Test content extraction handles network errors"""
        mock_get.side_effect = Exception("Network error")

        with self.assertRaises(Exception):
            self.content_extractor.extract_content_from_url("http://example.com")

    def test_validate_extraction_short_content(self):
        """Test validation with short content"""
        book_content = BookContent(
            id="test_id",
            url="http://example.com",
            title="Test",
            content="Short"
        )

        is_valid = self.content_extractor.validate_extraction(book_content)
        self.assertFalse(is_valid)

    def test_validate_extraction_valid_content(self):
        """Test validation with valid content"""
        book_content = BookContent(
            id="test_id",
            url="http://example.com",
            title="Test Title",
            content="This is a valid content with sufficient length to pass validation."
        )

        is_valid = self.content_extractor.validate_extraction(book_content)
        self.assertTrue(is_valid)


if __name__ == '__main__':
    unittest.main()