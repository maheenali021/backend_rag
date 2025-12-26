"""
Shared API response models for the Reusable Intelligence System
Defines common response structures for subagent interactions
"""
from typing import Dict, Any, List, Optional
from datetime import datetime


class BaseApiResponse:
    """
    Base class for API responses in the Reusable Intelligence System
    """

    def __init__(self, success: bool, message: str = ""):
        self.success = success
        self.message = message
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the response to a dictionary representation

        Returns:
            Dictionary representation of the response
        """
        return {
            "success": self.success,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }


class SubagentResponse(BaseApiResponse):
    """
    Response class for subagent execution results
    """

    def __init__(self, success: bool, result: Dict[str, Any], message: str = ""):
        super().__init__(success, message)
        self.result = result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the subagent response to a dictionary representation

        Returns:
            Dictionary representation of the subagent response
        """
        base_dict = super().to_dict()
        base_dict.update({
            "result": self.result
        })
        return base_dict


class ChapterDraftResponse(SubagentResponse):
    """
    Response class for chapter author subagent results
    """

    def __init__(self, success: bool, chapter_draft: Dict[str, Any], message: str = ""):
        super().__init__(success, {"chapter_draft": chapter_draft}, message)
        self.chapter_draft = chapter_draft

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the chapter draft response to a dictionary representation

        Returns:
            Dictionary representation of the chapter draft response
        """
        base_dict = super().to_dict()
        base_dict["chapter_draft"] = self.chapter_draft
        return base_dict


class ReviewResultsResponse(SubagentResponse):
    """
    Response class for technical reviewer subagent results
    """

    def __init__(self, success: bool, review_results: Dict[str, Any], message: str = ""):
        super().__init__(success, {"review_results": review_results}, message)
        self.review_results = review_results

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the review results response to a dictionary representation

        Returns:
            Dictionary representation of the review results response
        """
        base_dict = super().to_dict()
        base_dict["review_results"] = self.review_results
        return base_dict


class SummaryResultResponse(SubagentResponse):
    """
    Response class for summarizer subagent results
    """

    def __init__(self, success: bool, summary_result: Dict[str, Any], message: str = ""):
        super().__init__(success, {"result": summary_result}, message)
        self.summary_result = summary_result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the summary result response to a dictionary representation

        Returns:
            Dictionary representation of the summary result response
        """
        base_dict = super().to_dict()
        base_dict["result"] = self.summary_result
        return base_dict


class ErrorResponse(BaseApiResponse):
    """
    Response class for error situations
    """

    def __init__(self, message: str, error_code: str = "GENERIC_ERROR"):
        super().__init__(success=False, message=message)
        self.error_code = error_code

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error response to a dictionary representation

        Returns:
            Dictionary representation of the error response
        """
        base_dict = super().to_dict()
        base_dict.update({
            "error_code": self.error_code
        })
        return base_dict


class ValidationFeedback:
    """
    Class for representing feedback from validation processes
    """

    def __init__(self, feedback_type: str, category: str, message: str, location: str = "",
                 severity: str = "medium"):
        self.type = feedback_type  # "error", "warning", "suggestion", "positive"
        self.category = category  # "technical", "style", "structure", "accuracy", "completeness"
        self.message = message
        self.location = location
        self.severity = severity  # "critical", "high", "medium", "low"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the validation feedback to a dictionary representation

        Returns:
            Dictionary representation of the validation feedback
        """
        return {
            "type": self.type,
            "category": self.category,
            "message": self.message,
            "location": self.location,
            "severity": self.severity
        }


class SkillExecutionResponse(BaseApiResponse):
    """
    Response class for agent skill execution results
    """

    def __init__(self, success: bool, skill_result: Dict[str, Any], message: str = ""):
        super().__init__(success, message)
        self.skill_result = skill_result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the skill execution response to a dictionary representation

        Returns:
            Dictionary representation of the skill execution response
        """
        base_dict = super().to_dict()
        base_dict.update({
            "skill_result": self.skill_result
        })
        return base_dict


class WorkflowResponse(BaseApiResponse):
    """
    Response class for workflow execution results
    """

    def __init__(self, success: bool, workflow_steps: List[Dict[str, Any]], message: str = ""):
        super().__init__(success, message)
        self.workflow_steps = workflow_steps

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the workflow response to a dictionary representation

        Returns:
            Dictionary representation of the workflow response
        """
        base_dict = super().to_dict()
        base_dict.update({
            "workflow_steps": self.workflow_steps
        })
        return base_dict