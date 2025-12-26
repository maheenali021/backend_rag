"""
Base Subagent class for the Reusable Intelligence System
Defines the common interface and functionality for all subagents
"""
import abc
import logging
from typing import Any, Dict, Optional
from datetime import datetime


class Subagent(abc.ABC):
    """
    Abstract base class for all subagents in the Reusable Intelligence System
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.logger = logging.getLogger(self.__class__.__name__)

    @abc.abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the subagent with the given input data

        Args:
            input_data: Dictionary containing the input parameters for the subagent

        Returns:
            Dictionary containing the results of the subagent execution
        """
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate the input data before execution

        Args:
            input_data: Dictionary containing the input parameters for the subagent

        Returns:
            Boolean indicating whether the input is valid
        """
        # Basic validation - input_data should be a dictionary
        if not isinstance(input_data, dict):
            self.logger.error("Input data must be a dictionary")
            return False
        return True

    def log_execution(self, input_data: Dict[str, Any], result: Dict[str, Any]):
        """
        Log the execution of the subagent

        Args:
            input_data: Input data that was processed
            result: Result that was returned
        """
        self.logger.info(f"Executed {self.name} with input keys: {list(input_data.keys())}")
        self.logger.debug(f"Result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")