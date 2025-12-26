"""
PromptTemplate system for the Reusable Intelligence System
Ensures deterministic and reusable behavior across subagent invocations
"""
import re
from typing import List, Dict, Any, Optional
from datetime import datetime


class PromptTemplate:
    """
    Class representing a structured template that ensures deterministic and reusable behavior
    across subagent invocations
    """

    def __init__(self, name: str, description: str, content: str, subagent_type: str):
        self.id = f"pt_{name.replace(' ', '_').lower()}"
        self.name = name
        self.description = description
        self.content = content
        self.subagent_type = subagent_type
        self.variables = self._extract_variables(content)
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        # Validate the prompt template
        self._validate()

    def _extract_variables(self, content: str) -> List[str]:
        """
        Extract all variables from the prompt template content
        Variables are expected to be in the format {variable_name} or {{variable_name}}

        Args:
            content: The prompt template content

        Returns:
            List of variable names found in the content
        """
        # Look for variables in the format {var} or {{var}}
        pattern = r'\{(\w+)\}'  # Matches {var} format
        variables = re.findall(pattern, content)

        # Remove duplicates while preserving order
        seen = set()
        unique_variables = []
        for var in variables:
            if var not in seen:
                seen.add(var)
                unique_variables.append(var)

        return unique_variables

    def _validate(self):
        """
        Validate the prompt template according to the defined rules
        """
        # Validate name length
        if not (3 <= len(self.name) <= 50):
            raise ValueError(f"Name must be 3-50 characters, got {len(self.name)}")

        # Validate content length
        if not (20 <= len(self.content) <= 5000):
            raise ValueError(f"Content must be 20-5000 characters, got {len(self.content)}")

        # Validate subagent type
        valid_types = ["chapter_writer", "content_reviewer", "summarizer"]
        if self.subagent_type not in valid_types:
            raise ValueError(f"Subagent type must be one of {valid_types}, got {self.subagent_type}")

        # Validate variables match placeholders in content
        for var in self.variables:
            if f"{{{var}}}" not in self.content and f"{{{{{var}}}}}" not in self.content:
                raise ValueError(f"Variable {var} does not match any placeholder in content")

    def render(self, **kwargs) -> str:
        """
        Render the prompt template by substituting variables with provided values

        Args:
            **kwargs: Keyword arguments where keys are variable names and values are their substitutions

        Returns:
            The rendered prompt string with variables substituted
        """
        rendered_content = self.content

        # Replace each variable with its provided value
        for var in self.variables:
            if var in kwargs:
                value = kwargs[var]
                # Replace both {var} and {{var}} formats
                rendered_content = rendered_content.replace(f"{{{var}}}", str(value))
                rendered_content = rendered_content.replace(f"{{{{{var}}}}}", str(value))
            else:
                # If a required variable is not provided, raise an error
                raise ValueError(f"Required variable '{var}' not provided for prompt template '{self.name}'")

        return rendered_content

    def update_content(self, new_content: str):
        """
        Update the content of the prompt template

        Args:
            new_content: New content for the prompt template
        """
        self.content = new_content
        self.variables = self._extract_variables(new_content)
        self.updated_at = datetime.now()

        # Re-validate after update
        self._validate()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the prompt template to a dictionary representation

        Returns:
            Dictionary representation of the prompt template
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "subagent_type": self.subagent_type,
            "variables": self.variables,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """
        Create a PromptTemplate instance from a dictionary

        Args:
            data: Dictionary containing prompt template data

        Returns:
            PromptTemplate instance
        """
        return cls(
            name=data["name"],
            description=data["description"],
            content=data["content"],
            subagent_type=data["subagent_type"]
        )


class PromptTemplateManager:
    """
    Manager class for handling multiple prompt templates
    """

    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}

    def register_template(self, template: PromptTemplate):
        """
        Register a prompt template

        Args:
            template: PromptTemplate instance to register
        """
        self.templates[template.id] = template

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template by ID

        Args:
            template_id: ID of the template to retrieve

        Returns:
            PromptTemplate instance or None if not found
        """
        return self.templates.get(template_id)

    def get_template_by_name(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template by name

        Args:
            name: Name of the template to retrieve

        Returns:
            PromptTemplate instance or None if not found
        """
        for template in self.templates.values():
            if template.name == name:
                return template
        return None

    def get_templates_by_type(self, subagent_type: str) -> List[PromptTemplate]:
        """
        Get all prompt templates for a specific subagent type

        Args:
            subagent_type: Type of subagent to get templates for

        Returns:
            List of PromptTemplate instances for the specified type
        """
        return [template for template in self.templates.values()
                if template.subagent_type == subagent_type]