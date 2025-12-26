"""
AgentInteraction model for the Reusable Intelligence System
Records interactions with subagents for tracking and analysis
"""
from typing import Dict, Any, Optional
from datetime import datetime


class AgentInteraction:
    """
    Class representing a record of an interaction with a subagent
    """

    def __init__(self, subagent_id: str, input_data: Dict[str, Any], output_data: Dict[str, Any],
                 context: str = "", user_id: Optional[str] = None):
        self.id = f"ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{subagent_id[:8]}"
        self.subagent_id = subagent_id
        self.input_data = input_data
        self.output_data = output_data
        self.context = context
        self.user_id = user_id
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the agent interaction to a dictionary representation

        Returns:
            Dictionary representation of the agent interaction
        """
        return {
            "id": self.id,
            "subagent_id": self.subagent_id,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "context": self.context,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentInteraction':
        """
        Create an AgentInteraction instance from a dictionary

        Args:
            data: Dictionary containing agent interaction data

        Returns:
            AgentInteraction instance
        """
        return cls(
            subagent_id=data["subagent_id"],
            input_data=data["input_data"],
            output_data=data["output_data"],
            context=data.get("context", ""),
            user_id=data.get("user_id")
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the interaction

        Returns:
            Dictionary with key information about the interaction
        """
        return {
            "id": self.id,
            "subagent_id": self.subagent_id,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "input_keys": list(self.input_data.keys()) if isinstance(self.input_data, dict) else [],
            "output_keys": list(self.output_data.keys()) if isinstance(self.output_data, dict) else [],
        }


class InteractionStore:
    """
    In-memory store for agent interactions
    In a production environment, this would be replaced with a database
    """

    def __init__(self):
        self.interactions: Dict[str, AgentInteraction] = {}

    def record_interaction(self, interaction: AgentInteraction):
        """
        Record an agent interaction

        Args:
            interaction: AgentInteraction instance to record
        """
        self.interactions[interaction.id] = interaction

    def get_interaction(self, interaction_id: str) -> Optional[AgentInteraction]:
        """
        Get an agent interaction by ID

        Args:
            interaction_id: ID of the interaction to retrieve

        Returns:
            AgentInteraction instance or None if not found
        """
        return self.interactions.get(interaction_id)

    def get_interactions_by_subagent(self, subagent_id: str) -> list[AgentInteraction]:
        """
        Get all interactions for a specific subagent

        Args:
            subagent_id: ID of the subagent to get interactions for

        Returns:
            List of AgentInteraction instances for the specified subagent
        """
        return [interaction for interaction in self.interactions.values()
                if interaction.subagent_id == subagent_id]

    def get_interactions_by_user(self, user_id: str) -> list[AgentInteraction]:
        """
        Get all interactions initiated by a specific user

        Args:
            user_id: ID of the user to get interactions for

        Returns:
            List of AgentInteraction instances initiated by the specified user
        """
        return [interaction for interaction in self.interactions.values()
                if interaction.user_id == user_id]

    def get_recent_interactions(self, limit: int = 10) -> list[AgentInteraction]:
        """
        Get the most recent interactions

        Args:
            limit: Maximum number of interactions to return

        Returns:
            List of the most recent AgentInteraction instances
        """
        sorted_interactions = sorted(self.interactions.values(),
                                   key=lambda x: x.timestamp, reverse=True)
        return sorted_interactions[:limit]

    def get_interactions_by_context(self, context: str) -> list[AgentInteraction]:
        """
        Get all interactions with a specific context

        Args:
            context: Context to search for

        Returns:
            List of AgentInteraction instances with the specified context
        """
        return [interaction for interaction in self.interactions.values()
                if context.lower() in interaction.context.lower()]