"""
Content extraction service for RAG Pipeline
Extracts content from the deployed website using web scraping
"""
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import time
from ..models.book_content import BookContent
from ..utils.logging_config import get_logger
from ..utils.exceptions import ContentExtractionError
from ..utils.helpers import sanitize_text, generate_id
from ..config import Config


class ContentExtractor:
    """
    Service class for extracting content from the deployed book website
    """
    def __init__(self):
        self.logger = get_logger()
        self.config = Config
        self.session = requests.Session()
        # Set a user agent to avoid being blocked by some sites
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def extract_content_from_url(self, url: str) -> BookContent:
        """
        Extract content from a single URL

        Args:
            url: URL to extract content from

        Returns:
            BookContent object with extracted content
        """
        try:
            self.logger.info(f"Extracting content from URL: {url}")

            response = self.session.get(url)
            response.raise_for_status()

            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else urlparse(url).path.split('/')[-1] or "Untitled"

            # Extract main content - for Docusaurus sites, look for main content areas
            # This is tailored for Docusaurus-generated sites
            main_content_selectors = [
                'main div[class*="docItemContainer"]',  # Docusaurus doc container
                'article',  # Standard article tag
                'main',  # Main content area
                'div[class*="container"]',  # General container
                'div[class*="content"]',  # Content divs
                '.main-wrapper',  # Common wrapper classes
                '.docs-content',  # Docusaurus specific
                '.theme-doc-content',  # Docusaurus theme content
                '.markdown',  # Markdown rendered content
            ]

            content_text = ""
            for selector in main_content_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        content_text += element.get_text(separator=' ', strip=True) + "\n\n"
                    break

            # If no specific content found, use the body
            if not content_text.strip():
                body = soup.find('body')
                if body:
                    # Remove navigation, headers, footers, and sidebars
                    for element in body.select('nav, header, footer, .navbar, .menu, .sidebar, .table-of-contents'):
                        element.decompose()
                    content_text = body.get_text(separator=' ', strip=True)

            # Sanitize the extracted content
            content_text = sanitize_text(content_text)

            # Create BookContent object
            book_content = BookContent(
                id=generate_id(url, prefix="bc_"),
                url=url,
                title=sanitize_text(title),
                content=content_text,
                metadata={
                    "extracted_from": url,
                    "extraction_timestamp": time.time(),
                    "content_length": len(content_text),
                    "content_type": "web_page"
                }
            )

            self.logger.info(f"Successfully extracted content from {url}, length: {len(content_text)} characters")
            return book_content

        except requests.RequestException as e:
            self.logger.error(f"Network error while extracting content from {url}: {str(e)}")
            raise ContentExtractionError(f"Network error while extracting content from {url}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            raise ContentExtractionError(f"Error extracting content from {url}: {str(e)}")

    def extract_content_from_website(self, base_url: str = None) -> List[BookContent]:
        """
        Extract content from the entire website by finding all relevant URLs

        Args:
            base_url: Base URL of the website (optional, defaults to config)

        Returns:
            List of BookContent objects
        """
        website_url = base_url or self.config.WEBSITE_URL
        self.logger.info(f"Starting website content extraction from: {website_url}")

        # First, extract content from the main URL
        all_content = []

        try:
            # Extract main page content
            main_content = self.extract_content_from_url(website_url)
            all_content.append(main_content)

            # Get links from the main page to find other pages
            response = self.session.get(website_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all internal links
            base_domain = urlparse(website_url).netloc
            links = set()

            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(website_url, href)

                # Only include links from the same domain
                if urlparse(full_url).netloc == base_domain:
                    # Filter for likely content pages (not assets, external links, etc.)
                    if (full_url.startswith(website_url) and
                        not any(full_url.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.css', '.js', '.zip', '.exe'])):
                        links.add(full_url)

            self.logger.info(f"Found {len(links)} internal links to process")

            # Extract content from each link
            for link in links:
                try:
                    # Add a small delay to be respectful to the server
                    time.sleep(0.5)

                    content = self.extract_content_from_url(link)
                    all_content.append(content)

                except ContentExtractionError as e:
                    self.logger.warning(f"Failed to extract content from {link}: {str(e)}")
                    continue  # Continue with other URLs even if one fails

            self.logger.info(f"Successfully extracted content from {len(all_content)} pages")
            return all_content

        except Exception as e:
            self.logger.error(f"Error during website content extraction: {str(e)}")
            raise ContentExtractionError(f"Error during website content extraction: {str(e)}")

    def extract_content_by_sitemap(self, sitemap_url: str = None) -> List[BookContent]:
        """
        Extract content using a sitemap if available

        Args:
            sitemap_url: URL of the sitemap (optional, defaults to base_url + /sitemap.xml)

        Returns:
            List of BookContent objects
        """
        website_url = self.config.WEBSITE_URL
        sitemap_url = sitemap_url or f"{website_url.rstrip('/')}/sitemap.xml"

        self.logger.info(f"Attempting to extract content using sitemap: {sitemap_url}")

        try:
            response = self.session.get(sitemap_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'xml')  # Use XML parser for sitemap

            urls = []
            for loc in soup.find_all('loc'):
                url = loc.text.strip()
                if url.startswith(website_url):  # Only include URLs from the same site
                    urls.append(url)

            self.logger.info(f"Found {len(urls)} URLs in sitemap")

            all_content = []
            for url in urls:
                try:
                    # Add a small delay to be respectful to the server
                    time.sleep(0.5)

                    content = self.extract_content_from_url(url)
                    all_content.append(content)

                except ContentExtractionError as e:
                    self.logger.warning(f"Failed to extract content from {url}: {str(e)}")
                    continue  # Continue with other URLs even if one fails

            self.logger.info(f"Successfully extracted content from {len(all_content)} pages using sitemap")
            return all_content

        except requests.HTTPError:
            self.logger.warning(f"Sitemap not found at {sitemap_url}, falling back to link extraction")
            # Fall back to link extraction method
            return self.extract_content_from_website()
        except Exception as e:
            self.logger.error(f"Error processing sitemap: {str(e)}")
            # Fall back to link extraction method
            return self.extract_content_from_website()

    def validate_extraction(self, book_content: BookContent) -> bool:
        """
        Validate the extracted content

        Args:
            book_content: BookContent object to validate

        Returns:
            True if content is valid, False otherwise
        """
        # Check if content is substantial
        if not book_content.content or len(book_content.content.strip()) < 50:
            self.logger.warning(f"Content for {book_content.id} is too short: {len(book_content.content)} characters")
            return False

        # Check if title is meaningful
        if not book_content.title or len(book_content.title.strip()) < 2:
            self.logger.warning(f"Title for {book_content.id} is too short: {book_content.title}")
            return False

        return True