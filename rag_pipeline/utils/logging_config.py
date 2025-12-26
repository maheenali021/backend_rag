"""
Logging configuration for RAG Pipeline
Sets up consistent logging across all modules
"""
import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_logging(log_level=logging.INFO, log_file='rag_pipeline.log'):
    """
    Set up logging configuration for the RAG pipeline

    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: rag_pipeline.log)
    """
    # Create logger
    logger = logging.getLogger('rag_pipeline')
    logger.setLevel(log_level)

    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Create file handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logging()


def get_logger():
    """Get the configured logger instance"""
    return logger