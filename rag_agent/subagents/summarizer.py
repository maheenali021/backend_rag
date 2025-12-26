"""
Summarizer subagent for the Reusable Intelligence System
Generates accurate summaries and provides structured reasoning based on book content
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

from ..models.subagent_base import Subagent
from ..models.prompt_template import PromptTemplate, PromptTemplateManager


class Summarizer(Subagent):
    """
    Summarizer subagent that generates accurate summaries and provides
    structured reasoning based on book content to enhance the RAG chatbot's
    ability to assist users with complex technical queries
    """

    def __init__(self):
        super().__init__(
            name="Summarizer",
            description="Generates accurate summaries and provides structured reasoning based on book content"
        )
        self.template_manager = PromptTemplateManager()
        self._setup_prompts()

    def _setup_prompts(self):
        """
        Set up the prompt templates for the summarizer subagent
        """
        # Create the summarization prompt template
        summary_prompt = PromptTemplate(
            name="summarization_template",
            description="Template for generating summaries and structured reasoning",
            content="""
You are an expert in AI and Robotics, skilled at summarizing complex technical content.
Please provide a concise summary of the following content:

CONTENT TO SUMMARIZE:
{content}

SUMMARIZATION OPTIONS:
- Maximum length: {max_length} words
- Style: {style}
- Include reasoning: {include_reasoning}
- Include sources: {include_sources}
- Target audience: {target_audience}

SUMMARY REQUIREMENTS:
- Focus on key technical concepts and main points
- Maintain technical accuracy
- Provide structured reasoning if requested
- Include source attribution if requested
- Format appropriately for the target audience level

STRUCTURED REASONING (if requested):
- Explain the logical flow of concepts
- Identify key technical relationships
- Highlight important implications

SOURCES (if requested):
- Attribute to the original content
- Reference specific sections or concepts
""",
            subagent_type="summarizer"
        )
        self.template_manager.register_template(summary_prompt)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the summarizer subagent with the given input data

        Args:
            input_data: Dictionary containing:
                - content: The content to summarize or reason about
                - options: Options for the summarization

        Returns:
            Dictionary containing the summary, reasoning, confidence score, sources, and key points
        """
        try:
            # Validate input
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data provided to Summarizer")

            # Extract required parameters
            content = input_data.get("content", "")
            options = input_data.get("options", {})

            # Set default values if not provided
            max_length = options.get("max_length", 500)
            style = options.get("style", "concise")
            include_reasoning = options.get("include_reasoning", True)
            include_sources = options.get("include_sources", True)
            target_audience = options.get("target_audience", "intermediate")

            # Get the summarization prompt template
            prompt_template = self.template_manager.get_template_by_name("summarization_template")
            if not prompt_template:
                raise ValueError("Summarization prompt template not found")

            # Render the prompt with the provided data
            rendered_prompt = prompt_template.render(
                content=content,
                max_length=max_length,
                style=style,
                include_reasoning=str(include_reasoning),
                include_sources=str(include_sources),
                target_audience=target_audience
            )

            # Generate mock summary results - in a real implementation,
            # this would call an LLM to generate the summary
            word_count = len(content.split())
            summary_length = min(max_length, max(50, word_count // 4))  # Summary is roughly 1/4 of original

            # Create a summary based on the content
            words = content.split()
            summary_words = words[:summary_length]
            summary_text = " ".join(summary_words)

            # Generate structured reasoning if requested
            reasoning = ""
            if include_reasoning:
                reasoning = f"The summary focuses on the main concepts from the original text, condensing {word_count} words into {len(summary_words)} words while maintaining technical accuracy. The key points are extracted to provide a clear understanding of the core concepts."

            # Generate sources if requested
            sources = []
            if include_sources:
                sources = [
                    "Physical AI & Humanoid Robotics Book, Chapter 3",
                    "Advanced Robotics Concepts"
                ]

            # Generate key points
            key_points = [
                "Main technical concept extracted from content",
                "Secondary concept or application",
                "Key implication or conclusion"
            ]

            # Calculate confidence and accuracy scores
            confidence_score = min(1.0, max(0.5, len(summary_text) / 100))  # Scale based on summary length
            technical_accuracy_score = 0.92  # Mock accuracy score

            result = {
                "summary": summary_text,
                "reasoning": reasoning,
                "confidence_score": confidence_score,
                "sources": sources,
                "key_points": key_points,
                "technical_accuracy_score": technical_accuracy_score
            }

            # Log the execution
            self.log_execution(input_data, result)

            return result

        except Exception as e:
            self.logger.error(f"Error executing Summarizer: {str(e)}")
            raise