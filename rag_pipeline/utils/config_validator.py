"""
Configuration validation utilities for RAG Pipeline
Validates configuration settings before pipeline execution
"""
from typing import Dict, List, Tuple
from ..config import Config
from ..utils.logging_config import get_logger
from ..utils.exceptions import ConfigurationError


class ConfigValidator:
    """
    Utility class for validating configuration settings
    """
    def __init__(self):
        self.logger = get_logger()

    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        Validate the current configuration

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Validate Cohere configuration
        cohere_errors = self._validate_cohere_config()
        errors.extend(cohere_errors)

        # Validate Qdrant configuration
        qdrant_errors = self._validate_qdrant_config()
        errors.extend(qdrant_errors)

        # Validate website URL
        website_errors = self._validate_website_config()
        errors.extend(website_errors)

        # Validate chunking parameters
        chunking_errors = self._validate_chunking_config()
        errors.extend(chunking_errors)

        is_valid = len(errors) == 0
        return is_valid, errors

    def _validate_cohere_config(self) -> List[str]:
        """
        Validate Cohere-related configuration

        Returns:
            List of validation errors
        """
        errors = []

        if not Config.COHERE_API_KEY:
            errors.append("COHERE_API_KEY is not set in configuration")

        if not Config.COHERE_MODEL:
            errors.append("COHERE_MODEL is not set in configuration")
        elif Config.COHERE_MODEL not in [
            "embed-multilingual-v3.0",
            "embed-english-v3.0",
            "embed-multilingual-v2.0",
            "embed-english-v2.0"
        ]:
            errors.append(f"Unknown Cohere model: {Config.COHERE_MODEL}")

        return errors

    def _validate_qdrant_config(self) -> List[str]:
        """
        Validate Qdrant-related configuration

        Returns:
            List of validation errors
        """
        errors = []

        if not Config.QDRANT_URL:
            errors.append("QDRANT_URL is not set in configuration")

        if not Config.QDRANT_API_KEY:
            errors.append("QDRANT_API_KEY is not set in configuration")

        if not Config.QDRANT_COLLECTION_NAME:
            errors.append("QDRANT_COLLECTION_NAME is not set in configuration")

        return errors

    def _validate_website_config(self) -> List[str]:
        """
        Validate website-related configuration

        Returns:
            List of validation errors
        """
        errors = []

        if not Config.WEBSITE_URL:
            errors.append("WEBSITE_URL is not set in configuration")
        else:
            from urllib.parse import urlparse
            try:
                result = urlparse(Config.WEBSITE_URL)
                if not all([result.scheme, result.netloc]):
                    errors.append(f"Invalid WEBSITE_URL format: {Config.WEBSITE_URL}")
            except Exception as e:
                errors.append(f"Invalid WEBSITE_URL: {str(e)}")

        return errors

    def _validate_chunking_config(self) -> List[str]:
        """
        Validate chunking-related configuration

        Returns:
            List of validation errors
        """
        errors = []

        if Config.CHUNK_SIZE <= 0:
            errors.append(f"CHUNK_SIZE must be positive, got: {Config.CHUNK_SIZE}")

        if Config.CHUNK_OVERLAP < 0:
            errors.append(f"CHUNK_OVERLAP must be non-negative, got: {Config.CHUNK_OVERLAP}")

        if Config.CHUNK_OVERLAP >= Config.CHUNK_SIZE:
            errors.append(f"CHUNK_OVERLAP ({Config.CHUNK_OVERLAP}) should be less than CHUNK_SIZE ({Config.CHUNK_SIZE})")

        return errors

    def validate_for_pipeline(self) -> bool:
        """
        Validate configuration specifically for pipeline execution

        Returns:
            True if configuration is valid for pipeline execution
        """
        is_valid, errors = self.validate_config()

        if not is_valid:
            for error in errors:
                self.logger.error(error)
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")

        self.logger.info("Configuration validation passed")
        return True

    def get_config_summary(self) -> Dict[str, str]:
        """
        Get a summary of the current configuration (with sensitive data masked)

        Returns:
            Dictionary with configuration summary
        """
        return {
            "cohere_model": Config.COHERE_MODEL,
            "qdrant_collection": Config.QDRANT_COLLECTION_NAME,
            "website_url": Config.WEBSITE_URL,
            "chunk_size": str(Config.CHUNK_SIZE),
            "chunk_overlap": str(Config.CHUNK_OVERLAP),
            "cohere_api_key_set": bool(Config.COHERE_API_KEY),
            "qdrant_api_key_set": bool(Config.QDRANT_API_KEY),
            "qdrant_url": Config.QDRANT_URL if Config.QDRANT_URL else "Not set"
        }