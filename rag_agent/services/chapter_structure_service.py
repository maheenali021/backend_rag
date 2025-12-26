"""
Chapter Structure Service for the Reusable Intelligence System
Provides functionality for creating and managing structured chapter layouts
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime


class ChapterStructureService:
    """
    Service class for creating and managing structured chapter layouts
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_chapter_structure(self, topic: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a structured chapter layout based on the topic and requirements

        Args:
            topic: The main topic for the chapter
            requirements: Specific requirements for the chapter structure

        Returns:
            Dictionary containing the chapter structure with title, sections, and metadata
        """
        try:
            self.logger.info(f"Creating chapter structure for topic: {topic}")

            # Extract requirements
            target_audience = requirements.get("target_audience", "intermediate")
            length = requirements.get("length", 2000)
            sections = requirements.get("sections", [])
            examples_needed = requirements.get("examples_needed", True)

            # Create default sections if none provided
            if not sections:
                sections = self._get_default_sections(topic, target_audience)

            # Create the chapter structure
            chapter_structure = {
                "title": topic,
                "target_audience": target_audience,
                "estimated_length": length,
                "sections": self._create_section_outlines(sections, topic, target_audience),
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "examples_included": examples_needed,
                    "difficulty_level": self._map_audience_to_difficulty(target_audience)
                }
            }

            self.logger.info(f"Chapter structure created successfully for topic: {topic}")
            return chapter_structure

        except Exception as e:
            self.logger.error(f"Error creating chapter structure: {str(e)}")
            raise

    def _get_default_sections(self, topic: str, target_audience: str) -> List[str]:
        """
        Get default sections based on the topic and target audience

        Args:
            topic: The main topic for the chapter
            target_audience: The target audience level

        Returns:
            List of default section titles
        """
        base_sections = ["Introduction", "Main Content", "Examples", "Summary"]

        # Adjust sections based on topic
        if "algorithm" in topic.lower() or "method" in topic.lower():
            base_sections = ["Introduction", "Algorithm Overview", "Implementation", "Examples", "Complexity Analysis", "Summary"]
        elif "architecture" in topic.lower() or "design" in topic.lower():
            base_sections = ["Introduction", "Architecture Overview", "Components", "Design Patterns", "Implementation", "Best Practices", "Summary"]
        elif "humanoid" in topic.lower() or "robot" in topic.lower():
            base_sections = ["Introduction", "Background", "Technical Specifications", "Control Systems", "Applications", "Challenges", "Future Directions", "Summary"]

        return base_sections

    def _create_section_outlines(self, sections: List[str], topic: str, target_audience: str) -> List[Dict[str, Any]]:
        """
        Create outlines for each section based on the topic and target audience

        Args:
            sections: List of section titles
            topic: The main topic for the chapter
            target_audience: The target audience level

        Returns:
            List of dictionaries containing section outlines
        """
        section_outlines = []

        for section_title in sections:
            outline = {
                "title": section_title,
                "focus": self._get_section_focus(section_title, topic),
                "content_guidelines": self._get_content_guidelines(section_title, target_audience),
                "estimated_word_count": self._estimate_section_length(section_title, target_audience),
                "technical_depth": self._get_technical_depth(target_audience)
            }
            section_outlines.append(outline)

        return section_outlines

    def _get_section_focus(self, section_title: str, topic: str) -> str:
        """
        Get the focus for a section based on the title and topic

        Args:
            section_title: The title of the section
            topic: The main topic for the chapter

        Returns:
            Focus statement for the section
        """
        section_lower = section_title.lower()

        if "intro" in section_lower:
            return f"Introduce the concept of {topic} and its relevance"
        elif "overview" in section_lower or "background" in section_lower:
            return f"Provide background information and context for {topic}"
        elif "implement" in section_lower or "design" in section_lower:
            return f"Detail the implementation or design aspects of {topic}"
        elif "example" in section_lower:
            return f"Demonstrate {topic} with practical examples"
        elif "summary" in section_lower or "conclus" in section_lower:
            return f"Summarize key points about {topic} and potential future directions"
        elif "challeng" in section_lower:
            return f"Discuss challenges and limitations related to {topic}"
        else:
            return f"Cover aspects of {topic} relevant to this section"

    def _get_content_guidelines(self, section_title: str, target_audience: str) -> List[str]:
        """
        Get content guidelines for a section based on the title and target audience

        Args:
            section_title: The title of the section
            target_audience: The target audience level

        Returns:
            List of content guidelines for the section
        """
        guidelines = []

        # Add audience-specific guidelines
        if target_audience == "beginner":
            guidelines.extend([
                "Use simple language and avoid jargon",
                "Provide clear definitions for technical terms",
                "Include plenty of examples and analogies",
                "Explain the 'why' behind concepts, not just the 'what'"
            ])
        elif target_audience == "intermediate":
            guidelines.extend([
                "Use appropriate technical terminology",
                "Provide balanced explanation of concepts and implementation",
                "Include some practical examples",
                "Reference related concepts in the book"
            ])
        elif target_audience == "advanced":
            guidelines.extend([
                "Use advanced technical terminology appropriately",
                "Focus on implementation details and nuances",
                "Include complex examples and edge cases",
                "Reference cutting-edge research and developments"
            ])

        # Add section-specific guidelines
        section_lower = section_title.lower()
        if "intro" in section_lower:
            guidelines.extend([
                "Hook the reader with relevance of the topic",
                "Provide a roadmap of what the chapter covers",
                "Establish context for the chapter's content"
            ])
        elif "summary" in section_lower:
            guidelines.extend([
                "Recap main points covered in the chapter",
                "Highlight key takeaways",
                "Suggest next steps or related topics"
            ])

        return guidelines

    def _estimate_section_length(self, section_title: str, target_audience: str) -> int:
        """
        Estimate the word count for a section based on the title and target audience

        Args:
            section_title: The title of the section
            target_audience: The target audience level

        Returns:
            Estimated word count for the section
        """
        base_length = 200  # Default length

        section_lower = section_title.lower()
        if "intro" in section_lower or "summary" in section_lower:
            base_length = 150  # Shorter for intro/summary
        elif "implement" in section_lower or "detail" in section_lower:
            base_length = 300  # Longer for implementation details
        elif "example" in section_lower:
            base_length = 250  # Medium length for examples

        # Adjust based on audience
        if target_audience == "beginner":
            return int(base_length * 1.2)  # More explanation needed
        elif target_audience == "advanced":
            return int(base_length * 0.8)  # More concise for advanced audience
        else:  # intermediate
            return base_length

    def _get_technical_depth(self, target_audience: str) -> str:
        """
        Get the technical depth level based on the target audience

        Args:
            target_audience: The target audience level

        Returns:
            Technical depth level
        """
        if target_audience == "beginner":
            return "basic"
        elif target_audience == "intermediate":
            return "moderate"
        else:  # advanced
            return "deep"

    def _map_audience_to_difficulty(self, target_audience: str) -> str:
        """
        Map the target audience to a difficulty level

        Args:
            target_audience: The target audience level

        Returns:
            Difficulty level
        """
        audience_to_difficulty = {
            "beginner": "easy",
            "intermediate": "medium",
            "advanced": "hard"
        }
        return audience_to_difficulty.get(target_audience, "medium")

    def refine_chapter_structure(self, existing_structure: Dict[str, Any], enhancements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine an existing chapter structure with enhancements

        Args:
            existing_structure: The existing chapter structure to refine
            enhancements: Dictionary containing enhancement requests

        Returns:
            Refined chapter structure
        """
        try:
            self.logger.info(f"Refining chapter structure for: {existing_structure['title']}")

            # Make a copy of the existing structure
            refined_structure = existing_structure.copy()

            # Apply enhancements
            if "sections" in enhancements:
                refined_structure["sections"] = self._update_sections(
                    refined_structure["sections"],
                    enhancements["sections"]
                )

            if "target_audience" in enhancements:
                refined_structure["target_audience"] = enhancements["target_audience"]
                # Update section depths based on new audience
                for section in refined_structure["sections"]:
                    section["technical_depth"] = self._get_technical_depth(enhancements["target_audience"])

            if "add_examples" in enhancements and enhancements["add_examples"]:
                refined_structure = self._add_example_sections(refined_structure)

            self.logger.info(f"Chapter structure refined successfully")
            return refined_structure

        except Exception as e:
            self.logger.error(f"Error refining chapter structure: {str(e)}")
            raise

    def _update_sections(self, existing_sections: List[Dict[str, Any]], new_sections: List[str]) -> List[Dict[str, Any]]:
        """
        Update existing sections with new section titles

        Args:
            existing_sections: List of existing section dictionaries
            new_sections: List of new section titles

        Returns:
            Updated list of section dictionaries
        """
        # Create a mapping of existing section titles to their details
        existing_map = {section["title"]: section for section in existing_sections}

        updated_sections = []
        for section_title in new_sections:
            if section_title in existing_map:
                # Keep existing details but update title if needed
                updated_sections.append(existing_map[section_title])
            else:
                # Create new section with default properties
                new_section = {
                    "title": section_title,
                    "focus": f"Cover aspects of the topic relevant to {section_title}",
                    "content_guidelines": ["Include relevant content for this section"],
                    "estimated_word_count": 200,
                    "technical_depth": "moderate"
                }
                updated_sections.append(new_section)

        return updated_sections

    def _add_example_sections(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add example sections to the chapter structure

        Args:
            structure: The chapter structure to enhance

        Returns:
            Enhanced chapter structure with example sections
        """
        # Add example sections after main content sections
        enhanced_sections = []
        for section in structure["sections"]:
            enhanced_sections.append(section)
            # Add an example section after implementation/design sections
            if any(keyword in section["title"].lower() for keyword in ["implementation", "design", "method", "algorithm"]):
                example_section = {
                    "title": f"{section['title']} - Examples",
                    "focus": f"Provide practical examples of {section['title']}",
                    "content_guidelines": ["Include concrete examples", "Show real-world applications"],
                    "estimated_word_count": 150,
                    "technical_depth": section["technical_depth"]
                }
                enhanced_sections.append(example_section)

        structure["sections"] = enhanced_sections
        return structure