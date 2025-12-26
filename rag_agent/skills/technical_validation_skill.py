"""
Technical Validation Skill for the Reusable Intelligence System
Validates content for technical accuracy and consistency with book standards
"""
import logging
import re
from typing import Dict, Any, List
from datetime import datetime

from ..models.agent_skill import AgentSkill


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


class TechnicalValidationSkill(AgentSkill):
    """
    Agent skill for validating content for technical accuracy and consistency
    with book standards, identifying technical errors and suggesting improvements
    """

    def __init__(self):
        super().__init__(
            name="Technical Validation",
            description="Validates content for technical accuracy and consistency with book standards",
            category="validation"
        )
        # Define common technical terms and concepts for validation
        self.technical_terms = self._define_technical_terms()

    def _define_technical_terms(self) -> Dict[str, str]:
        """
        Define common technical terms and concepts for validation

        Returns:
            Dictionary mapping terms to their definitions for validation purposes
        """
        return {
            "AI": "Artificial Intelligence",
            "ML": "Machine Learning",
            "NN": "Neural Network",
            "CNN": "Convolutional Neural Network",
            "RNN": "Recurrent Neural Network",
            "LSTM": "Long Short-Term Memory",
            "RL": "Reinforcement Learning",
            "NLP": "Natural Language Processing",
            "CV": "Computer Vision",
            "DL": "Deep Learning",
            "ROS": "Robot Operating System",
            "SLAM": "Simultaneous Localization and Mapping",
            "PID": "Proportional-Integral-Derivative",
            "IK": "Inverse Kinematics",
            "FK": "Forward Kinematics",
            "DOF": "Degrees of Freedom",
            "URDF": "Unified Robot Description Format",
            "Gazebo": "Robot simulation environment",
            "OpenCV": "Open Source Computer Vision Library",
            "TensorFlow": "Machine learning framework",
            "PyTorch": "Machine learning framework",
            "Q-learning": "Reinforcement learning algorithm",
            "Monte Carlo": "Simulation method",
            "Kalman Filter": "Estimation algorithm",
            "Point Cloud": "3D data representation",
            "LiDAR": "Light Detection and Ranging",
            "IMU": "Inertial Measurement Unit",
            "Actuator": "Device that produces motion",
            "Sensor": "Device that detects changes",
            "Controller": "Device that manages system behavior",
            "Trajectory": "Path through space over time",
            "End Effector": "Robot tool at manipulator end",
            "Workspace": "Volume robot end effector can reach",
            "Configuration Space": "Space of robot joint positions",
            "Collision Detection": "Process of detecting collisions",
            "Path Planning": "Finding path from start to goal",
            "Motion Planning": "Planning robot movement",
            "Task Planning": "Planning high-level robot tasks",
            "Humanoid Robot": "Robot with human-like form",
            "Bipedal Locomotion": "Two-legged walking",
            "Balance Control": "Maintaining robot stability",
            "Whole Body Control": "Controlling entire robot body",
            "Force Control": "Controlling forces applied by robot",
            "Impedance Control": "Controlling robot's mechanical impedance",
            "Admittance Control": "Controlling robot's motion in response to forces",
            "Visual Servoing": "Controlling robot using visual feedback",
            "Teaching by Demonstration": "Learning from human demonstration",
            "Behavior Trees": "Hierarchical task execution model",
            "Finite State Machines": "Computational model with finite states",
            "Bayesian Networks": "Probabilistic graphical model",
            "Monte Carlo Localization": "Robot localization technique",
            "Particle Filters": "Probabilistic estimation technique",
            "Computer Vision": "Field of computer science dealing with visual data",
            "Pattern Recognition": "Identifying patterns in data",
            "Machine Learning": "Method of training computers to learn from data",
            "Deep Learning": "Subset of ML using neural networks",
            "Supervised Learning": "Learning with labeled training data",
            "Unsupervised Learning": "Learning without labeled training data",
            "Reinforcement Learning": "Learning through reward/punishment",
            "Transfer Learning": "Applying knowledge from one domain to another",
            "Few-shot Learning": "Learning with few examples",
            "Zero-shot Learning": "Learning without examples",
            "Embodied AI": "AI integrated into physical systems",
            "Physical AI": "AI for physical world interactions",
            "Robotics": "Science of robot design and operation"
        }

    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        Define the input schema for the technical validation skill
        """
        return {
            "type": "object",
            "required": ["content"],
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to validate for technical accuracy",
                    "minLength": 10,
                    "maxLength": 50000
                },
                "criteria": {
                    "type": "object",
                    "description": "Specific criteria for the validation",
                    "properties": {
                        "technical_accuracy": {
                            "type": "boolean",
                            "default": True,
                            "description": "Validate for technical accuracy"
                        },
                        "style_consistency": {
                            "type": "boolean",
                            "default": True,
                            "description": "Validate for style consistency with book standards"
                        },
                        "target_audience": {
                            "type": "string",
                            "enum": ["beginner", "intermediate", "advanced"],
                            "default": "intermediate",
                            "description": "Target audience level for appropriateness"
                        },
                        "completeness": {
                            "type": "boolean",
                            "default": True,
                            "description": "Validate for completeness of coverage"
                        }
                    }
                }
            }
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        """
        Define the output schema for the technical validation skill
        """
        return {
            "type": "object",
            "required": ["validation_results"],
            "properties": {
                "validation_results": {
                    "type": "object",
                    "properties": {
                        "feedback": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["error", "warning", "suggestion", "positive"]
                                    },
                                    "category": {
                                        "type": "string",
                                        "enum": ["technical", "style", "structure", "accuracy", "completeness"]
                                    },
                                    "message": {
                                        "type": "string",
                                        "description": "Detailed feedback message"
                                    },
                                    "location": {
                                        "type": "string",
                                        "description": "Location in the content where issue occurs"
                                    },
                                    "severity": {
                                        "type": "string",
                                        "enum": ["critical", "high", "medium", "low"]
                                    }
                                }
                            }
                        },
                        "accuracy_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Overall technical accuracy score"
                        },
                        "suggestions": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Specific improvement suggestions"
                        },
                        "summary": {
                            "type": "object",
                            "properties": {
                                "technical_issues": {
                                    "type": "integer",
                                    "description": "Number of technical issues found"
                                },
                                "style_issues": {
                                    "type": "integer",
                                    "description": "Number of style issues found"
                                },
                                "suggestions_count": {
                                    "type": "integer",
                                    "description": "Number of suggestions provided"
                                }
                            }
                        }
                    }
                }
            }
        }

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the technical validation skill with the given input data

        Args:
            input_data: Dictionary containing:
                - content: The content to validate
                - criteria: Specific criteria for the validation

        Returns:
            Dictionary containing validation results with feedback, accuracy score, and suggestions
        """
        try:
            # Validate input against schema
            if not self.validate_input(input_data):
                raise ValueError("Invalid input data provided to Technical Validation Skill")

            # Extract parameters
            content = input_data.get("content", "")
            criteria = input_data.get("criteria", {})

            # Set default values
            technical_accuracy = criteria.get("technical_accuracy", True)
            style_consistency = criteria.get("style_consistency", True)
            target_audience = criteria.get("target_audience", "intermediate")
            completeness = criteria.get("completeness", True)

            # Perform validation based on criteria
            feedback = []
            suggestions = []

            if technical_accuracy:
                feedback.extend(self._validate_technical_accuracy(content))

            if style_consistency:
                feedback.extend(self._validate_style_consistency(content))

            if completeness:
                feedback.extend(self._validate_completeness(content))

            # Adjust feedback based on target audience
            feedback = self._adjust_feedback_for_audience(feedback, target_audience)

            # Calculate accuracy score
            accuracy_score = self._calculate_accuracy_score(feedback, len(content))

            # Generate suggestions based on feedback
            suggestions = self._generate_suggestions(feedback)

            # Create summary
            summary = self._create_validation_summary(feedback, suggestions)

            result = {
                "validation_results": {
                    "feedback": [item.to_dict() for item in feedback],
                    "accuracy_score": accuracy_score,
                    "suggestions": suggestions,
                    "summary": summary
                }
            }

            # Log the execution
            self.log_execution(input_data, result)

            return result

        except Exception as e:
            self.logger.error(f"Error executing Technical Validation Skill: {str(e)}")
            raise

    def _validate_technical_accuracy(self, content: str) -> List[ValidationFeedback]:
        """
        Validate the content for technical accuracy

        Args:
            content: The content to validate

        Returns:
            List of validation feedback items
        """
        feedback = []

        # Check for undefined technical terms
        content_lower = content.lower()
        for term, definition in self.technical_terms.items():
            if term.lower() in content_lower:
                # Check if the term is properly defined in the content
                if len(content) < 500:  # Very short content might not define terms
                    continue

                # Look for common patterns where terms should be defined
                if term.upper() == term and len(term) > 1:  # Acronym
                    # Check if the acronym is defined (e.g., "AI (Artificial Intelligence)")
                    acronym_pattern = rf"{term}\s*\([^)]*\)"
                    if not re.search(acronym_pattern, content, re.IGNORECASE):
                        feedback.append(ValidationFeedback(
                            "warning",
                            "technical",
                            f"Acronym '{term}' should be defined in full on first use (e.g., 'Term ({definition})')",
                            "acronym definition",
                            "medium"
                        ))

        # Check for common technical mistakes
        if re.search(r"(AI is|AI are)\s+.*\b(?:algorithm|program|software)\b", content, re.IGNORECASE):
            feedback.append(ValidationFeedback(
                "error",
                "technical",
                "AI should be described as 'systems' or 'technologies' rather than 'algorithms' or 'programs'",
                "AI terminology",
                "high"
            ))

        # Check for technical inconsistencies
        if re.search(r"(machine learning|ML).*accuracy.*100%", content, re.IGNORECASE):
            feedback.append(ValidationFeedback(
                "warning",
                "technical",
                "Claiming 100% accuracy for machine learning models is unrealistic; real-world models rarely achieve perfect accuracy",
                "accuracy claim",
                "high"
            ))

        # Check for technical plausibility
        if re.search(r"(quantum computer|quantum computing).*solve.*NP-complete", content, re.IGNORECASE):
            feedback.append(ValidationFeedback(
                "error",
                "technical",
                "Quantum computers are not known to solve NP-complete problems efficiently; this is a common misconception",
                "quantum computing claim",
                "high"
            ))

        return feedback

    def _validate_style_consistency(self, content: str) -> List[ValidationFeedback]:
        """
        Validate the content for style consistency with book standards

        Args:
            content: The content to validate

        Returns:
            List of validation feedback items
        """
        feedback = []

        # Check for inconsistent terminology
        if "neural network" in content.lower() and "neural net" in content.lower():
            feedback.append(ValidationFeedback(
                "warning",
                "style",
                "Inconsistent terminology: use either 'neural network' or 'neural net' consistently throughout",
                "terminology consistency",
                "medium"
            ))

        # Check for passive vs active voice consistency
        passive_voice_count = len(re.findall(r"\b(am|is|are|was|were|been|being)\b\s+\w+ed\b", content, re.IGNORECASE))
        if passive_voice_count > 10:  # Arbitrary threshold
            feedback.append(ValidationFeedback(
                "suggestion",
                "style",
                "High use of passive voice detected; consider using more active voice for clarity",
                "voice consistency",
                "medium"
            ))

        # Check for formatting consistency (headings, lists, etc.)
        heading_count = len(re.findall(r"^#{1,6}\s+", content, re.MULTILINE))
        if heading_count > 0 and not re.search(r"^#{1,6}\s+.+\n", content, re.MULTILINE):
            feedback.append(ValidationFeedback(
                "warning",
                "structure",
                "Headings detected but may not follow consistent formatting pattern",
                "heading structure",
                "low"
            ))

        return feedback

    def _validate_completeness(self, content: str) -> List[ValidationFeedback]:
        """
        Validate the content for completeness of coverage

        Args:
            content: The content to validate

        Returns:
            List of validation feedback items
        """
        feedback = []

        # Check if content is too brief for a comprehensive treatment
        word_count = len(content.split())
        if word_count < 300:
            feedback.append(ValidationFeedback(
                "warning",
                "completeness",
                f"Content appears brief ({word_count} words); consider expanding with more technical details, examples, or applications",
                "content length",
                "medium"
            ))

        # Check for common missing elements in technical content
        content_lower = content.lower()

        # Check if important concepts are missing
        if "algorithm" in content_lower and "complexity" not in content_lower:
            feedback.append(ValidationFeedback(
                "suggestion",
                "completeness",
                "Algorithm discussed but computational complexity not mentioned; consider adding time/space complexity analysis",
                "complexity analysis",
                "medium"
            ))

        if "method" in content_lower and "evaluation" not in content_lower and "performance" not in content_lower:
            feedback.append(ValidationFeedback(
                "suggestion",
                "completeness",
                "Method discussed but evaluation/performance not mentioned; consider adding evaluation metrics or benchmarks",
                "performance evaluation",
                "medium"
            ))

        return feedback

    def _adjust_feedback_for_audience(self, feedback: List[ValidationFeedback], target_audience: str) -> List[ValidationFeedback]:
        """
        Adjust feedback based on the target audience level

        Args:
            feedback: Original feedback list
            target_audience: Target audience level

        Returns:
            Adjusted feedback list
        """
        adjusted_feedback = []

        for item in feedback:
            # Adjust severity based on audience
            if target_audience == "beginner" and item.category == "technical" and item.severity == "high":
                # For beginners, technical errors should be flagged more prominently
                item.severity = "critical"
            elif target_audience == "advanced" and item.category == "style":
                # For advanced audiences, style issues might be less critical
                if item.severity == "low":
                    item.severity = "medium"
                elif item.severity == "medium":
                    item.severity = "low"

            # Add context about audience appropriateness
            if target_audience == "beginner" and "mathematical" in item.message.lower():
                item.message += f" (Consider simplifying for {target_audience} audience)"

            adjusted_feedback.append(item)

        return adjusted_feedback

    def _calculate_accuracy_score(self, feedback: List[ValidationFeedback], content_length: int) -> float:
        """
        Calculate an overall technical accuracy score based on feedback

        Args:
            feedback: List of validation feedback items
            content_length: Length of the content being validated

        Returns:
            Accuracy score between 0 and 1
        """
        if content_length == 0:
            return 0.0

        # Start with a perfect score
        score = 1.0

        # Deduct points for different types of issues
        for item in feedback:
            if item.type == "error":
                if item.severity == "critical":
                    score -= 0.3
                elif item.severity == "high":
                    score -= 0.15
                elif item.severity == "medium":
                    score -= 0.05
                elif item.severity == "low":
                    score -= 0.01
            elif item.type == "warning":
                if item.severity == "high":
                    score -= 0.1
                elif item.severity == "medium":
                    score -= 0.05
                elif item.severity == "low":
                    score -= 0.02

        # Adjust for content length (longer content has more opportunity for errors)
        length_factor = min(1.0, content_length / 2000.0)  # Normalize based on 2000 word baseline
        score = score * (0.8 + 0.2 * length_factor)  # Give some leniency for longer content

        # Ensure score stays within bounds
        return max(0.0, min(1.0, score))

    def _generate_suggestions(self, feedback: List[ValidationFeedback]) -> List[str]:
        """
        Generate improvement suggestions based on feedback

        Args:
            feedback: List of validation feedback items

        Returns:
            List of improvement suggestions
        """
        suggestions = set()  # Use set to avoid duplicates

        for item in feedback:
            if item.type == "error":
                suggestions.add(f"Correct the technical error: {item.message}")
            elif item.type == "warning":
                suggestions.add(f"Address the potential issue: {item.message}")
            elif item.type == "suggestion":
                suggestions.add(item.message)
            elif item.type == "positive":
                # Don't add positive items as suggestions for improvement
                continue

        # Add general suggestions based on feedback patterns
        error_count = sum(1 for f in feedback if f.type == "error")
        if error_count > 3:
            suggestions.add("Consider having this content reviewed by a technical expert before publication")

        technical_issue_count = sum(1 for f in feedback if f.category == "technical")
        if technical_issue_count > 2:
            suggestions.add("Verify all technical claims with authoritative sources")

        return list(suggestions)

    def _create_validation_summary(self, feedback: List[ValidationFeedback], suggestions: List[str]) -> Dict[str, int]:
        """
        Create a summary of the validation results

        Args:
            feedback: List of validation feedback items
            suggestions: List of improvement suggestions

        Returns:
            Summary dictionary with counts
        """
        technical_issues = sum(1 for f in feedback if f.category == "technical")
        style_issues = sum(1 for f in feedback if f.category == "style")
        structure_issues = sum(1 for f in feedback if f.category == "structure")
        accuracy_issues = sum(1 for f in feedback if f.category == "accuracy")
        completeness_issues = sum(1 for f in feedback if f.category == "completeness")

        return {
            "technical_issues": technical_issues,
            "style_issues": style_issues,
            "structure_issues": structure_issues,
            "accuracy_issues": accuracy_issues,
            "completeness_issues": completeness_issues,
            "suggestions_count": len(suggestions)
        }