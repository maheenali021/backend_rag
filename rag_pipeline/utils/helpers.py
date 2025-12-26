"""
Utility functions for RAG Pipeline
Contains common helper functions used across different modules
"""
import time
import hashlib
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse


def generate_id(content: str, prefix: str = "") -> str:
    """
    Generate a unique ID based on content hash

    Args:
        content: Content to hash
        prefix: Optional prefix for the ID

    Returns:
        Unique ID string
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{prefix}{content_hash}" if prefix else content_hash


def sanitize_text(text: str) -> str:
    """
    Sanitize text by removing extra whitespace and normalizing

    Args:
        text: Input text to sanitize

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Replace multiple whitespace with single space
    sanitized = ' '.join(text.split())
    return sanitized.strip()


def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL

    Args:
        url: URL string to validate

    Returns:
        True if valid URL, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def batch_list(lst: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a list into batches of specified size

    Args:
        lst: List to batch
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    if not lst:
        return []

    batches = []
    for i in range(0, len(lst), batch_size):
        batches.append(lst[i:i + batch_size])

    return batches


def retry_with_backoff(func, max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Execute a function with exponential backoff retry logic

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        backoff_factor: Factor for exponential backoff

    Returns:
        Result of function execution
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = backoff_factor * (2 ** attempt)
                time.sleep(wait_time)
            else:
                raise last_exception

    raise last_exception


def normalize_url(url: str) -> str:
    """
    Normalize URL by ensuring it has proper scheme and removing fragments

    Args:
        url: URL to normalize

    Returns:
        Normalized URL
    """
    parsed = urlparse(url)

    # Ensure scheme is present
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)

    # Reconstruct URL without fragment
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        normalized += f"?{parsed.query}"

    return normalized


def format_bytes(size_bytes: int) -> str:
    """
    Format bytes to human readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Human readable size string
    """
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.2f}{size_names[i]}"