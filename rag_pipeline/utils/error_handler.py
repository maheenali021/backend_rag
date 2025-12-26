"""
Error handling middleware for RAG Pipeline
Provides consistent error handling across all modules
"""
import traceback
from typing import Callable, Any, Optional
from functools import wraps
from ..utils.logging_config import get_logger
from ..utils.exceptions import RAGPipelineError


class ErrorHandler:
    """
    Error handling utility class
    Provides methods for consistent error handling and logging
    """
    def __init__(self):
        self.logger = get_logger()

    def handle_error(self, error: Exception, context: str = "", reraise: bool = True):
        """
        Handle an error by logging it and optionally re-raising

        Args:
            error: The exception to handle
            context: Context information about where the error occurred
            reraise: Whether to re-raise the exception after logging

        Returns:
            None, but may raise the original exception if reraise is True
        """
        error_msg = f"Error in {context}: {str(error)}"
        error_type = type(error).__name__

        # Log the full traceback
        self.logger.error(f"{error_type}: {error_msg}")
        self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")

        if reraise:
            raise error

    def safe_execute(self, func: Callable, *args, default_return: Any = None, **kwargs):
        """
        Safely execute a function, catching and logging any exceptions

        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            default_return: Default value to return if an exception occurs
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Function result or default_return if an exception occurs
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {func.__name__}: {str(e)}")
            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            return default_return

    def retry_on_failure(self, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
        """
        Decorator to retry a function on failure

        Args:
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries
            backoff: Multiplier for delay on each retry
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                current_delay = delay
                last_exception = None

                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries:
                            self.logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                                f"Retrying in {current_delay} seconds..."
                            )
                            import time
                            time.sleep(current_delay)
                            current_delay *= backoff
                        else:
                            self.logger.error(
                                f"All {max_retries + 1} attempts failed for {func.__name__}. "
                                f"Last error: {str(e)}"
                            )
                            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")

                # If we've exhausted all retries, raise the last exception
                raise last_exception
            return wrapper
        return decorator

    def validate_and_handle(self, validation_func: Callable, error_func: Callable, *args, **kwargs):
        """
        Validate input using validation_func, handle errors with error_func if validation fails

        Args:
            validation_func: Function that performs validation and raises exception if invalid
            error_func: Function to handle validation errors
            *args: Arguments to pass to validation_func
            **kwargs: Keyword arguments to pass to validation_func

        Returns:
            Result of validation_func if successful, result of error_func if validation fails
        """
        try:
            return validation_func(*args, **kwargs)
        except Exception as e:
            return error_func(e)

    def log_exception(self, exception: Exception, level: str = "error", extra_info: Optional[dict] = None):
        """
        Log an exception with additional context

        Args:
            exception: Exception to log
            level: Logging level ('debug', 'info', 'warning', 'error', 'critical')
            extra_info: Additional information to include in the log
        """
        logger_method = getattr(self.logger, level, self.logger.error)

        if extra_info:
            logger_method(f"Exception: {str(exception)}, Extra info: {extra_info}")
        else:
            logger_method(f"Exception: {str(exception)}")

        # Always log the full traceback at debug level
        self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")


# Global error handler instance
error_handler = ErrorHandler()


def handle_pipeline_error(context: str = "", reraise: bool = True):
    """
    Decorator to handle errors in pipeline functions

    Args:
        context: Context information about where the error occurred
        reraise: Whether to re-raise the exception after logging
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RAGPipelineError:
                # Re-raise RAGPipeline-specific errors directly
                raise
            except Exception as e:
                error_handler.handle_error(e, f"{context or func.__name__}", reraise)
        return wrapper
    return decorator


def safe_pipeline_operation(default_return: Any = None):
    """
    Decorator to safely execute pipeline operations

    Args:
        default_return: Default value to return if an exception occurs
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return error_handler.safe_execute(func, *args, default_return=default_return, **kwargs)
        return wrapper
    return decorator