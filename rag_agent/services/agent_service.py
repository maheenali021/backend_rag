"""
Agent Service for the Reusable Intelligence System
Orchestrates the interaction between different subagents and manages the overall workflow
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..subagents.chapter_author import ChapterAuthor
from ..subagents.technical_reviewer import TechnicalReviewer
from ..subagents.summarizer import Summarizer
from ..models.chapter_draft import ChapterDraft
from ..models.review_results import ReviewResults
from ..models.summary_result import SummaryResult


class AgentService:
    """
    Main service class that orchestrates the interaction between different subagents
    and manages the overall workflow for content creation and validation
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.chapter_author = ChapterAuthor()
        self.technical_reviewer = TechnicalReviewer()
        self.summarizer = Summarizer()
        self.workflow_history = []

    async def create_chapter_with_review(self, topic: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a chapter with integrated review process

        Args:
            topic: The main topic for the chapter
            requirements: Specific requirements for the chapter

        Returns:
            Dictionary containing the chapter draft and review results
        """
        try:
            self.logger.info(f"Starting chapter creation process for topic: {topic}")

            # Step 1: Generate initial chapter draft
            self.logger.info("Step 1: Generating initial chapter draft")
            chapter_input = {
                "topic": topic,
                "requirements": requirements
            }

            chapter_draft = await self.chapter_author.execute(chapter_input)
            self.logger.info(f"Chapter draft generated with title: {chapter_draft.get('title', 'Unknown')}")

            # Step 2: Review the chapter draft
            self.logger.info("Step 2: Reviewing chapter draft")
            review_input = {
                "content": chapter_draft.get("content", ""),
                "criteria": {
                    "technical_accuracy": True,
                    "style_consistency": True,
                    "target_audience": requirements.get("target_audience", "intermediate"),
                    "completeness": True
                }
            }

            review_results = await self.technical_reviewer.execute(review_input)
            self.logger.info("Chapter draft reviewed successfully")

            # Step 3: Generate summary of the chapter
            self.logger.info("Step 3: Generating chapter summary")
            summary_input = {
                "content": chapter_draft.get("content", ""),
                "options": {
                    "max_length": 300,
                    "style": "technical",
                    "include_reasoning": True,
                    "include_sources": True,
                    "target_audience": requirements.get("target_audience", "intermediate")
                }
            }

            summary_result = await self.summarizer.execute(summary_input)
            self.logger.info("Chapter summary generated successfully")

            # Compile results
            result = {
                "chapter_draft": chapter_draft,
                "review_results": review_results,
                "summary_result": summary_result,
                "process_completed_at": datetime.now().isoformat(),
                "workflow_steps": [
                    {"step": "chapter_author", "status": "completed", "timestamp": chapter_draft.get("created_at")},
                    {"step": "technical_reviewer", "status": "completed", "timestamp": datetime.now().isoformat()},
                    {"step": "summarizer", "status": "completed", "timestamp": datetime.now().isoformat()}
                ]
            }

            # Add to workflow history
            self.workflow_history.append({
                "process": "chapter_creation_with_review",
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "result_ids": {
                    "chapter": chapter_draft.get("id"),
                    "review": review_results.get("id"),
                    "summary": summary_result.get("id")
                }
            })

            self.logger.info(f"Chapter creation process completed successfully for topic: {topic}")
            return result

        except Exception as e:
            self.logger.error(f"Error in chapter creation workflow: {str(e)}")
            raise

    async def process_content_batch(self, content_list: list, process_type: str = "review") -> list:
        """
        Process a batch of content items using the appropriate subagent

        Args:
            content_list: List of content items to process
            process_type: Type of processing ('review', 'summarize', 'generate')

        Returns:
            List of processed results
        """
        try:
            self.logger.info(f"Starting batch processing of {len(content_list)} items with type: {process_type}")

            results = []
            for i, content in enumerate(content_list):
                self.logger.info(f"Processing item {i+1}/{len(content_list)}")

                if process_type == "review":
                    review_input = {
                        "content": content,
                        "criteria": {
                            "technical_accuracy": True,
                            "style_consistency": True,
                            "target_audience": "intermediate",
                            "completeness": True
                        }
                    }
                    result = await self.technical_reviewer.execute(review_input)
                elif process_type == "summarize":
                    summary_input = {
                        "content": content,
                        "options": {
                            "max_length": 500,
                            "style": "concise",
                            "include_reasoning": True,
                            "include_sources": True,
                            "target_audience": "intermediate"
                        }
                    }
                    result = await self.summarizer.execute(summary_input)
                elif process_type == "generate":
                    # For generation, we need a topic instead of content
                    generation_input = {
                        "topic": content,
                        "requirements": {
                            "target_audience": "intermediate",
                            "length": 1000,
                            "examples_needed": True
                        }
                    }
                    result = await self.chapter_author.execute(generation_input)
                else:
                    raise ValueError(f"Unsupported process type: {process_type}")

                results.append(result)

            self.logger.info(f"Batch processing completed for {len(content_list)} items")
            return results

        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            raise

    async def run_validation_pipeline(self, content: str, validation_levels: list = None) -> Dict[str, Any]:
        """
        Run a comprehensive validation pipeline on content

        Args:
            content: Content to validate
            validation_levels: List of validation levels to run

        Returns:
            Dictionary containing validation results at different levels
        """
        if validation_levels is None:
            validation_levels = ["technical", "structural", "stylistic"]

        try:
            self.logger.info(f"Starting validation pipeline for content of length: {len(content)} characters")

            results = {
                "content_overview": {
                    "length": len(content),
                    "word_count": len(content.split()),
                    "timestamp": datetime.now().isoformat()
                },
                "validation_results": {}
            }

            for level in validation_levels:
                if level == "technical":
                    # Run technical validation
                    tech_validation_input = {
                        "content": content,
                        "criteria": {
                            "technical_accuracy": True,
                            "completeness": False,
                            "style_consistency": False
                        }
                    }
                    tech_result = await self.technical_reviewer.execute(tech_validation_input)
                    results["validation_results"]["technical"] = tech_result

                elif level == "structural":
                    # Run structural validation (could involve checking for proper sectioning, etc.)
                    # For now, this is a placeholder that returns basic structure information
                    structure_result = {
                        "section_count": content.count("# "),  # Rough count of sections
                        "paragraph_count": len(content.split('\n\n')),
                        "has_introduction": "introduction" in content.lower() or "intro" in content.lower(),
                        "has_conclusion": "conclusion" in content.lower() or "summary" in content.lower(),
                        "timestamp": datetime.now().isoformat()
                    }
                    results["validation_results"]["structural"] = structure_result

                elif level == "stylistic":
                    # Run stylistic validation
                    style_validation_input = {
                        "content": content,
                        "criteria": {
                            "technical_accuracy": False,
                            "completeness": False,
                            "style_consistency": True,
                            "target_audience": "intermediate"
                        }
                    }
                    style_result = await self.technical_reviewer.execute(style_validation_input)
                    results["validation_results"]["stylistic"] = style_result

            self.logger.info("Validation pipeline completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Error in validation pipeline: {str(e)}")
            raise

    def get_workflow_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the workflow execution history

        Returns:
            Dictionary containing workflow statistics
        """
        if not self.workflow_history:
            return {"message": "No workflow history available"}

        total_workflows = len(self.workflow_history)
        workflow_types = {}
        daily_activity = {}

        for entry in self.workflow_history:
            # Count workflow types
            wf_type = entry["process"]
            workflow_types[wf_type] = workflow_types.get(wf_type, 0) + 1

            # Track daily activity
            date = entry["timestamp"][:10]  # Extract YYYY-MM-DD
            daily_activity[date] = daily_activity.get(date, 0) + 1

        stats = {
            "total_workflows_executed": total_workflows,
            "workflow_types": workflow_types,
            "daily_activity": daily_activity,
            "most_common_workflow": max(workflow_types, key=workflow_types.get) if workflow_types else None,
            "recent_activity": self.workflow_history[-5:] if len(self.workflow_history) >= 5 else self.workflow_history
        }

        return stats

    async def run_multi_agent_collaboration(self, task_description: str, agents_to_use: list) -> Dict[str, Any]:
        """
        Run a collaborative task involving multiple agents

        Args:
            task_description: Description of the task to be performed
            agents_to_use: List of agent names to collaborate ('author', 'reviewer', 'summarizer')

        Returns:
            Dictionary containing collaborative results
        """
        try:
            self.logger.info(f"Starting multi-agent collaboration for task: {task_description}")

            results = {
                "task_description": task_description,
                "agents_involved": agents_to_use,
                "collaboration_steps": [],
                "final_output": {},
                "timestamp": datetime.now().isoformat()
            }

            current_content = task_description  # Start with the task description as initial content

            for agent_name in agents_to_use:
                step_result = {
                    "agent": agent_name,
                    "input_content_length": len(current_content),
                    "timestamp": datetime.now().isoformat()
                }

                if agent_name == "author":
                    # Generate content based on the task
                    author_input = {
                        "topic": task_description,
                        "requirements": {
                            "target_audience": "intermediate",
                            "length": 1500,
                            "examples_needed": True
                        }
                    }
                    author_result = await self.chapter_author.execute(author_input)
                    current_content = author_result.get("content", current_content)
                    step_result["output"] = author_result

                elif agent_name == "reviewer":
                    # Review the current content
                    review_input = {
                        "content": current_content,
                        "criteria": {
                            "technical_accuracy": True,
                            "style_consistency": True,
                            "target_audience": "intermediate",
                            "completeness": True
                        }
                    }
                    review_result = await self.technical_reviewer.execute(review_input)
                    step_result["output"] = review_result

                    # Update content based on feedback if needed
                    # In a real implementation, we might apply suggestions to the content

                elif agent_name == "summarizer":
                    # Summarize the current content
                    summary_input = {
                        "content": current_content,
                        "options": {
                            "max_length": 500,
                            "style": "technical",
                            "include_reasoning": True,
                            "include_sources": True,
                            "target_audience": "intermediate"
                        }
                    }
                    summary_result = await self.summarizer.execute(summary_input)
                    step_result["output"] = summary_result

                results["collaboration_steps"].append(step_result)

            # Set the final output to the last agent's output
            if results["collaboration_steps"]:
                results["final_output"] = results["collaboration_steps"][-1]["output"]

            self.logger.info(f"Multi-agent collaboration completed for task: {task_description}")
            return results

        except Exception as e:
            self.logger.error(f"Error in multi-agent collaboration: {str(e)}")
            raise

    def reset_workflow_history(self):
        """
        Reset the workflow execution history
        """
        self.workflow_history = []
        self.logger.info("Workflow history has been reset")

    async def run_content_enhancement_pipeline(self, initial_content: str, enhancement_types: list) -> Dict[str, Any]:
        """
        Run a pipeline to enhance content through multiple transformations

        Args:
            initial_content: Starting content to enhance
            enhancement_types: List of enhancement types to apply

        Returns:
            Dictionary containing the enhanced content and transformation details
        """
        try:
            self.logger.info(f"Starting content enhancement pipeline with types: {enhancement_types}")

            current_content = initial_content
            enhancement_log = []

            for enhancement_type in enhancement_types:
                original_length = len(current_content)
                enhancement_step = {
                    "enhancement_type": enhancement_type,
                    "input_length": original_length,
                    "timestamp": datetime.now().isoformat()
                }

                if enhancement_type == "expand":
                    # Expand content by adding more detail
                    expansion_input = {
                        "topic": f"Expansion of: {current_content[:100]}...",
                        "requirements": {
                            "target_audience": "intermediate",
                            "length": original_length * 2,  # Aim to double the length
                            "examples_needed": True
                        }
                    }
                    expansion_result = await self.chapter_author.execute(expansion_input)
                    current_content = expansion_result.get("content", current_content)

                elif enhancement_type == "refine":
                    # Refine content through review and feedback application
                    review_input = {
                        "content": current_content,
                        "criteria": {
                            "technical_accuracy": True,
                            "style_consistency": True,
                            "target_audience": "intermediate",
                            "completeness": True
                        }
                    }
                    review_result = await self.technical_reviewer.execute(review_input)

                    # In a real implementation, we would apply the feedback to the content
                    # For now, we'll just return the feedback as part of the result
                    current_content = current_content  # Content remains the same for this example

                elif enhancement_type == "summarize":
                    # Create a more concise version
                    summary_input = {
                        "content": current_content,
                        "options": {
                            "max_length": min(original_length // 2, 500),  # Reduce by half or max 500 words
                            "style": "concise",
                            "include_reasoning": False,
                            "include_sources": False,
                            "target_audience": "intermediate"
                        }
                    }
                    summary_result = await self.summarizer.execute(summary_input)
                    current_content = summary_result.get("summary", current_content)

                enhancement_step.update({
                    "output_length": len(current_content),
                    "length_change": len(current_content) - original_length,
                    "result": summary_result if enhancement_type == "summarize" else
                             review_result if enhancement_type == "refine" else
                             expansion_result if enhancement_type == "expand" else {}
                })

                enhancement_log.append(enhancement_step)

            result = {
                "initial_content_length": len(initial_content),
                "final_content_length": len(current_content),
                "enhanced_content": current_content,
                "enhancement_log": enhancement_log,
                "total_transformations_applied": len(enhancement_types),
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info("Content enhancement pipeline completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error in content enhancement pipeline: {str(e)}")
            raise