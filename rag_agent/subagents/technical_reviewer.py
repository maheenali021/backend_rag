"""
Technical Reviewer subagent for the Reusable Intelligence System
Reviews content for technical accuracy and consistency with the book's standards
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

from ..models.subagent_base import Subagent
from ..models.prompt_template import PromptTemplate, PromptTemplateManager
from ..models.api_responses import ValidationFeedback


class TechnicalReviewer(Subagent):
    """
    Technical Reviewer subagent that reviews content for technical accuracy
    and consistency with the book's standards, validating technical concepts
    and suggesting improvements
    """

    def __init__(self):
        super().__init__(
            name="Technical Reviewer",
            description="Reviews content for technical accuracy and consistency with book standards"
        )
        self.template_manager = PromptTemplateManager()
        self._setup_prompts()

    def _setup_prompts(self):
        """
        Set up the prompt templates for the technical reviewer subagent
        """
        # Create the content review prompt template
        review_prompt = PromptTemplate(
            name="content_review_template",
            description="Template for reviewing content for technical accuracy",
            content="""
You are an expert technical reviewer specializing in AI and Robotics. Review the following content for technical accuracy and consistency:

CONTENT TO REVIEW:
{content}

REVIEW CRITERIA:
- Technical accuracy: {technical_accuracy}
- Style consistency: {style_consistency}
- Target audience: {target_audience}
- Completeness: {completeness}

Please provide detailed feedback on:
1. Technical accuracy - identify any technical errors or inaccuracies
2. Style consistency - ensure it follows the book's standards
3. Appropriateness for target audience level
4. Completeness of coverage
5. Suggestions for improvement

Return your feedback in a structured format with specific locations of issues.
""",
            subagent_type="content_reviewer"
        )
        self.template_manager.register_template(review_prompt)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the technical reviewer subagent with the given input data

        Args:
            input_data: Dictionary containing:
                - content: The content to review
                - criteria: Specific criteria for the review

        Returns:
            Dictionary containing review results with feedback, accuracy score, and suggestions
        """
        try:
            # Validate input
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data provided to Technical Reviewer")

            # Extract required parameters
            content = input_data.get("content", "")
            criteria = input_data.get("criteria", {})

            # Set default values if not provided
            technical_accuracy = criteria.get("technical_accuracy", True)
            style_consistency = criteria.get("style_consistency", True)
            target_audience = criteria.get("target_audience", "intermediate")
            completeness = criteria.get("completeness", True)

            # Get the content review prompt template
            prompt_template = self.template_manager.get_template_by_name("content_review_template")
            if not prompt_template:
                raise ValueError("Content review prompt template not found")

            # Render the prompt with the provided data
            rendered_prompt = prompt_template.render(
                content=content,
                technical_accuracy=str(technical_accuracy),
                style_consistency=str(style_consistency),
                target_audience=target_audience,
                completeness=str(completeness)
            )

            # Generate mock review results - in a real implementation,
            # this would call an LLM to analyze the content
            feedback: List[Dict[str, Any]] = []

            # Add some example feedback based on content analysis
            if len(content) < 50:
                feedback.append({
                    "type": "warning",
                    "category": "completeness",
                    "message": "Content appears to be very brief - consider expanding with more technical details",
                    "location": "entire content",
                    "severity": "medium"
                })

            if "algorithm" in content.lower() or "method" in content.lower():
                feedback.append({
                    "type": "suggestion",
                    "category": "technical",
                    "message": "Consider adding more specific examples of the algorithm/method in practice",
                    "location": "technical concepts section",
                    "severity": "medium"
                })

            # Calculate mock accuracy score based on content characteristics
            word_count = len(content.split())
            technical_accuracy_score = min(1.0, max(0.5, word_count / 1000))

            # Generate mock suggestions
            suggestions = [
                "Consider adding more technical examples",
                "Ensure terminology is consistent with book standards"
            ]

            review_results = {
                "feedback": feedback,
                "accuracy_score": technical_accuracy_score,
                "suggestions": suggestions,
                "summary": {
                    "technical_issues": sum(1 for f in feedback if f["category"] == "technical"),
                    "style_issues": sum(1 for f in feedback if f["category"] == "style"),
                    "suggestions_count": len(suggestions)
                }
            }

            # Log the execution
            self.log_execution(input_data, review_results)

            return review_results

        except Exception as e:
            self.logger.error(f"Error executing Technical Reviewer: {str(e)}")
            raise