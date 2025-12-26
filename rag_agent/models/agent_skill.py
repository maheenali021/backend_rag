"""
AgentSkill interface for the Reusable Intelligence System
Defines the common interface for all agent skills with clear input/output schemas
"""
import abc
import logging
from typing import Any, Dict, Optional
from datetime import datetime


class AgentSkill(abc.ABC):
    """
    Abstract base class for all agent skills in the Reusable Intelligence System
    Each skill should have a clear input/output schema for consistent behavior
    """

    def __init__(self, name: str, description: str, category: str):
        self.name = name
        self.description = description
        self.category = category
        self.created_at = datetime.now()
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abc.abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """
        Define the input schema for this skill
        This should be a JSON Schema describing the expected input parameters
        """
        pass

    @property
    @abc.abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """
        Define the output schema for this skill
        This should be a JSON Schema describing the expected output format
        """
        pass

    @abc.abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent skill with the given input data

        Args:
            input_data: Dictionary containing the input parameters for the skill

        Returns:
            Dictionary containing the results of the skill execution
        """
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate the input data against the skill's input schema

        Args:
            input_data: Dictionary containing the input parameters for the skill

        Returns:
            Boolean indicating whether the input is valid according to the schema
        """
        # Basic validation - input_data should be a dictionary
        if not isinstance(input_data, dict):
            self.logger.error(f"Input data for skill {self.name} must be a dictionary")
            return False
        return True

    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """
        Validate the output data against the skill's output schema

        Args:
            output_data: Dictionary containing the output from the skill

        Returns:
            Boolean indicating whether the output is valid according to the schema
        """
        # Basic validation - output_data should be a dictionary
        if not isinstance(output_data, dict):
            self.logger.error(f"Output data for skill {self.name} must be a dictionary")
            return False
        return True

    def log_execution(self, input_data: Dict[str, Any], result: Dict[str, Any]):
        """
        Log the execution of the agent skill

        Args:
            input_data: Input data that was processed
            result: Result that was returned
        """
        self.logger.info(f"Executed skill {self.name} in category {self.category}")
        self.logger.debug(f"Input keys: {list(input_data.keys())}")
        self.logger.debug(f"Result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")