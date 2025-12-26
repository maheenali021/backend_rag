"""
Content Generation Skill for the Reusable Intelligence System
Generates structured content based on provided parameters and requirements
"""
import logging
from typing import Dict, Any
from datetime import datetime

from ..models.agent_skill import AgentSkill


class ContentGenerationSkill(AgentSkill):
    """
    Agent skill for generating structured content based on provided parameters and requirements
    """

    def __init__(self):
        super().__init__(
            name="Content Generation",
            description="Generates structured content based on provided parameters and requirements",
            category="content_generation"
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        Define the input schema for the content generation skill
        """
        return {
            "type": "object",
            "required": ["topic", "requirements"],
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The main topic for content generation",
                    "minLength": 5,
                    "maxLength": 200
                },
                "requirements": {
                    "type": "object",
                    "description": "Specific requirements for the content",
                    "properties": {
                        "target_audience": {
                            "type": "string",
                            "enum": ["beginner", "intermediate", "advanced"],
                            "description": "Target audience level for the content",
                            "default": "intermediate"
                        },
                        "length": {
                            "type": "integer",
                            "minimum": 100,
                            "maximum": 10000,
                            "default": 1000,
                            "description": "Target length in words"
                        },
                        "sections": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "default": [],
                            "description": "Specific sections that should be covered"
                        },
                        "examples_needed": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to include examples"
                        },
                        "style": {
                            "type": "string",
                            "enum": ["technical", "educational", "narrative", "expository"],
                            "default": "educational",
                            "description": "Writing style for the content"
                        }
                    }
                },
                "context": {
                    "type": "string",
                    "description": "Additional context or constraints",
                    "maxLength": 1000
                }
            }
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        """
        Define the output schema for the content generation skill
        """
        return {
            "type": "object",
            "required": ["generated_content", "metadata"],
            "properties": {
                "generated_content": {
                    "type": "string",
                    "description": "The generated content in markdown format"
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "word_count": {
                            "type": "integer",
                            "description": "Number of words in the generated content"
                        },
                        "estimated_reading_time": {
                            "type": "integer",
                            "description": "Estimated reading time in minutes"
                        },
                        "target_audience": {
                            "type": "string",
                            "enum": ["beginner", "intermediate", "advanced"],
                            "description": "Audience level of the generated content"
                        },
                        "quality_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Estimated quality score of the generated content"
                        }
                    }
                },
                "sections_covered": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Sections that were included in the content"
                }
            }
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the content generation skill with the given input data

        Args:
            input_data: Dictionary containing:
                - topic: The main topic for content generation
                - requirements: Specific requirements for the content
                - context: Additional context or constraints

        Returns:
            Dictionary containing the generated content and metadata
        """
        try:
            # Validate input against schema
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data provided to Content Generation Skill")

            # Extract parameters
            topic = input_data.get("topic", "")
            requirements = input_data.get("requirements", {})
            context = input_data.get("context", "")

            # Set default values
            target_audience = requirements.get("target_audience", "intermediate")
            length = requirements.get("length", 1000)
            sections = requirements.get("sections", [])
            examples_needed = requirements.get("examples_needed", True)
            style = requirements.get("style", "educational")

            # Generate content based on parameters
            generated_content = self._generate_content(
                topic=topic,
                target_audience=target_audience,
                length=length,
                sections=sections,
                examples_needed=examples_needed,
                style=style,
                context=context
            )

            # Calculate metadata
            word_count = len(generated_content.split())
            estimated_reading_time = max(1, word_count // 200)  # 200 words per minute reading speed
            quality_score = self._calculate_quality_score(word_count, target_audience, examples_needed)

            # Determine sections covered
            sections_covered = sections if sections else self._infer_sections_from_topic(topic)

            result = {
                "generated_content": generated_content,
                "metadata": {
                    "word_count": word_count,
                    "estimated_reading_time": estimated_reading_time,
                    "target_audience": target_audience,
                    "quality_score": quality_score
                },
                "sections_covered": sections_covered
            }

            # Log the execution
            self.log_execution(input_data, result)

            return result

        except Exception as e:
            self.logger.error(f"Error executing Content Generation Skill: {str(e)}")
            raise

    def _generate_content(self, topic: str, target_audience: str, length: int, sections: list,
                         examples_needed: bool, style: str, context: str) -> str:
        """
        Generate content based on the provided parameters

        Args:
            topic: The main topic for content generation
            target_audience: The target audience level
            length: Target length in words
            sections: Specific sections to cover
            examples_needed: Whether to include examples
            style: Writing style for the content
            context: Additional context or constraints

        Returns:
            Generated content string
        """
        # For this implementation, we'll generate a mock content structure
        # In a real implementation, this would call an LLM to generate actual content

        content_parts = []

        # Add title
        content_parts.append(f"# {topic}\n")

        # Add introduction based on target audience
        intro_style = self._get_introduction_style(target_audience, style)
        content_parts.append(f"## Introduction\n{intro_style}\n")

        # Add content sections
        if sections:
            for section in sections:
                content_parts.append(f"## {section}\n{self._generate_section_content(section, target_audience, length//len(sections) if sections else length//3)}\n")
        else:
            # Default sections if none provided
            default_sections = ["Background", "Main Content", "Applications"]
            for section in default_sections:
                content_parts.append(f"## {section}\n{self._generate_section_content(section, target_audience, length//len(default_sections))}\n")

        # Add examples if needed
        if examples_needed:
            content_parts.append(f"## Examples\n{self._generate_examples(topic, target_audience)}\n")

        # Add conclusion
        content_parts.append(f"## Conclusion\n{self._generate_conclusion(topic, target_audience)}\n")

        # Add context if provided
        if context:
            content_parts.append(f"\n## Additional Context\n{context}\n")

        # Join all parts
        full_content = "\n".join(content_parts)

        # Truncate to approximate desired length
        words = full_content.split()
        if len(words) > length:
            words = words[:length]
            full_content = " ".join(words) + "..."

        return full_content

    def _get_introduction_style(self, target_audience: str, style: str) -> str:
        """
        Get the appropriate introduction style based on target audience and content style

        Args:
            target_audience: The target audience level
            style: Writing style for the content

        Returns:
            Introduction content string
        """
        if target_audience == "beginner":
            return f"This chapter introduces the concept of {style} in a beginner-friendly manner. We'll start with the basics and build up to more complex ideas, making sure each concept is clearly explained before moving on to the next."
        elif target_audience == "advanced":
            return f"This chapter delves deep into the advanced concepts of {style}. It assumes familiarity with the fundamentals and explores cutting-edge developments, nuanced implementations, and complex applications in the field."
        else:  # intermediate
            return f"This chapter explores the topic of {style}, building on fundamental concepts while introducing more sophisticated ideas. We'll cover both theoretical foundations and practical applications."

    def _generate_section_content(self, section: str, target_audience: str, section_length: int) -> str:
        """
        Generate content for a specific section

        Args:
            section: The section title
            target_audience: The target audience level
            section_length: Approximate length for this section

        Returns:
            Section content string
        """
        # Generate mock content for the section
        content = f"This section covers {section.lower()}, focusing on key concepts relevant to {target_audience} level understanding. "

        if target_audience == "beginner":
            content += "We'll explain fundamental principles with clear definitions and simple examples to build your understanding. "
        elif target_audience == "advanced":
            content += "We'll explore complex implementations, edge cases, and cutting-edge research in this area. "
        else:  # intermediate
            content += "We'll examine both foundational concepts and more sophisticated applications. "

        # Add more content based on section type
        if "background" in section.lower() or "history" in section.lower():
            content += "Historical context and evolution of these concepts provide insight into current approaches. "
        elif "application" in section.lower() or "use" in section.lower():
            content += "Real-world implementations demonstrate the practical value of these theoretical concepts. "
        elif "future" in section.lower() or "trend" in section.lower():
            content += "Emerging trends and future directions indicate the evolving landscape of this field. "

        # Extend content to approximate desired length
        while len(content.split()) < section_length:
            content += f"Additional details about {section.lower()} continue to build a comprehensive understanding. "

        return content

    def _generate_examples(self, topic: str, target_audience: str) -> str:
        """
        Generate examples for the content

        Args:
            topic: The main topic
            target_audience: The target audience level

        Returns:
            Examples content string
        """
        example_content = f"This section provides practical examples of {topic} in action. "

        if target_audience == "beginner":
            example_content += "Simple, concrete examples illustrate basic principles and help solidify understanding. "
        elif target_audience == "advanced":
            example_content += "Complex, real-world examples showcase sophisticated implementations and nuanced applications. "
        else:  # intermediate
            example_content += "Moderately complex examples bridge the gap between theory and practice. "

        example_content += f"Each example is carefully chosen to demonstrate key aspects of {topic}."

        return example_content

    def _generate_conclusion(self, topic: str, target_audience: str) -> str:
        """
        Generate conclusion for the content

        Args:
            topic: The main topic
            target_audience: The target audience level

        Returns:
            Conclusion content string
        """
        conclusion = f"In conclusion, we've explored the key aspects of {topic}. "

        if target_audience == "beginner":
            conclusion += "This introduction provides the foundation needed to explore more advanced concepts in this field. "
        elif target_audience == "advanced":
            conclusion += "These advanced concepts represent the current state of the art, with numerous avenues for further research and development. "
        else:  # intermediate
            conclusion += "This exploration provides a solid understanding of both fundamental and intermediate concepts. "

        conclusion += f"Future study of {topic} will build upon these foundations."

        return conclusion

    def _infer_sections_from_topic(self, topic: str) -> list:
        """
        Infer likely sections from the topic

        Args:
            topic: The main topic

        Returns:
            List of inferred sections
        """
        topic_lower = topic.lower()

        if "algorithm" in topic_lower or "method" in topic_lower:
            return ["Introduction", "Algorithm Overview", "Implementation", "Examples", "Complexity Analysis", "Summary"]
        elif "architecture" in topic_lower or "design" in topic_lower:
            return ["Introduction", "Architecture Overview", "Components", "Design Patterns", "Implementation", "Best Practices", "Summary"]
        elif "humanoid" in topic_lower or "robot" in topic_lower:
            return ["Introduction", "Background", "Technical Specifications", "Control Systems", "Applications", "Challenges", "Future Directions", "Summary"]
        else:
            return ["Introduction", "Main Concepts", "Applications", "Examples", "Summary"]

    def _calculate_quality_score(self, word_count: int, target_audience: str, examples_needed: bool) -> float:
        """
        Calculate a quality score based on content characteristics

        Args:
            word_count: Number of words in the content
            target_audience: The target audience level
            examples_needed: Whether examples were requested

        Returns:
            Quality score between 0 and 1
        """
        score = 0.5  # Base score

        # Adjust for word count (too short or too long reduces score)
        if 300 <= word_count <= 2000:
            score += 0.2
        elif 100 <= word_count <= 5000:
            score += 0.1

        # Adjust for audience appropriateness
        if target_audience in ["beginner", "intermediate", "advanced"]:
            score += 0.1

        # Adjust for presence of examples
        if examples_needed:
            score += 0.1

        return min(1.0, max(0.0, score))