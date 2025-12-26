"""
RAG Agent Service for the AI Robotics Textbook
Handles chat queries with RAG (Retrieval-Augmented Generation) functionality
"""
import logging
from typing import Dict, Any, Optional
import os
from datetime import datetime
import asyncio

from openai import OpenAI
from ..models.agent_models import AgentRequest, AgentResponse, RetrievedChunk, ConversationSession, QueryType
from ..config import Config
from ..tools.retrieval_tool import RetrievalTool
from ..utils.validation_utils import ResponseValidator


class RAGAgentService:
    """
    RAG Agent Service that handles chat queries with retrieval-augmented generation
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Force load environment variables at the module level to ensure they're available
        import os
        from dotenv import load_dotenv

        # Define the paths to the .env files
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(os.path.dirname(current_file_dir))
        project_root = os.path.dirname(backend_dir)
        pipeline_dir = os.path.join(backend_dir, 'rag_pipeline')
        rag_agent_dir = os.path.join(backend_dir, 'rag_agent')

        # Load environment variables from all relevant .env files with override=True
        env_files = [
            os.path.join(project_root, '.env'),
            os.path.join(pipeline_dir, '.env'),
            os.path.join(rag_agent_dir, '.env')
        ]

        for env_file in env_files:
            if os.path.isfile(env_file):
                load_dotenv(env_file, override=True)
                self.logger.info(f"Loaded environment variables from: {env_file}")

        # Now reload the config to pick up the loaded environment variables
        from rag_agent.config import Config
        self.config = Config

        # Validate configuration
        try:
            Config.validate()
            self.logger.info("Configuration validated successfully")
        except ValueError as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise

        # Initialize OpenRouter client
        try:
            # Ensure the API key is available at this point
            api_key = os.getenv('OPENROUTER_API_KEY') or self.config.OPENROUTER_API_KEY
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is not set")

            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            self.logger.info("OpenRouter client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenRouter client: {str(e)}")
            raise

        # Initialize retrieval tool
        try:
            self.retrieval_tool = RetrievalTool()
            self.logger.info("Retrieval tool initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize retrieval tool: {str(e)}")
            raise

        # In-memory session storage (in production, use Redis or database)
        self.sessions: Dict[str, ConversationSession] = {}

        self.logger.info("RAG Agent Service initialized successfully")

    def create_conversation_session(self) -> ConversationSession:
        """Create a new conversation session"""
        import uuid
        session_id = str(uuid.uuid4())
        session = ConversationSession(id=session_id)
        self.sessions[session_id] = session
        self.logger.info(f"Created new conversation session: {session_id}")
        return session

    def get_conversation_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a conversation session by ID"""
        session = self.sessions.get(session_id)
        if session:
            self.logger.info(f"Retrieved conversation session: {session_id}")
        else:
            self.logger.warning(f"Conversation session not found: {session_id}")
        return session

    def clear_conversation(self, session_id: str) -> bool:
        """Clear a conversation session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info(f"Cleared conversation session: {session_id}")
            return True
        return False

    def process_request(self, request: AgentRequest) -> AgentResponse:
        """
        Process an agent request and return a response
        """
        try:
            self.logger.info(f"Processing request: {request.query[:50]}... (type: {request.query_type})")

            # Generate conversation ID if not provided
            conversation_id = request.conversation_id or f"conv_{int(datetime.now().timestamp())}"

            # Perform retrieval based on query type
            retrieved_chunks = []
            source_attribution = set()

            if request.query_type == QueryType.GENERAL:
                # General query - search across all content
                filters = request.filters or {}
                retrieved_data = self.retrieval_tool.search(
                    query=request.query,
                    filters=filters,
                    top_k=request.top_k
                )

            elif request.query_type == QueryType.CHAPTER_SPECIFIC and request.chapter_filter:
                # Chapter-specific query
                filters = {"chapter": request.chapter_filter}
                if request.filters:
                    filters.update(request.filters)
                retrieved_data = self.retrieval_tool.search(
                    query=request.query,
                    filters=filters,
                    top_k=request.top_k
                )

            elif request.query_type == QueryType.USER_CONTEXT:
                # User context mode - no retrieval needed, but we can still search for related content
                retrieved_data = []
                self.logger.info("Processing in user context mode - no retrieval performed")
            else:
                # Default general search
                retrieved_data = self.retrieval_tool.search(
                    query=request.query,
                    top_k=request.top_k
                )

            # Convert retrieved data to RetrievedChunk objects
            for item in retrieved_data:
                chunk = RetrievedChunk(
                    id=item["id"],
                    content=item["content"],
                    source_url=item["source_url"],
                    chapter=item.get("chapter"),
                    section=item.get("section"),
                    similarity_score=item["similarity_score"],
                    confidence_score=item["confidence_score"],
                    retrieval_timestamp=datetime.fromisoformat(item["retrieval_timestamp"]) if isinstance(item["retrieval_timestamp"], str) else item["retrieval_timestamp"],
                    metadata=item.get("metadata", {})
                )
                retrieved_chunks.append(chunk)
                source_attribution.add(item["source_url"])

            # Build context from retrieved chunks
            context_text = ""
            if retrieved_chunks:
                context_text = "\n\n".join([f"Source: {chunk.source_url}\nContent: {chunk.content}" for chunk in retrieved_chunks])
            elif request.query_type == QueryType.USER_CONTEXT and request.user_context:
                # Use user-provided context
                context_text = f"User Context: {request.user_context}"

            # Prepare the prompt for the LLM
            if context_text:
                prompt = f"""
                You are an AI assistant for the AI Robotics textbook. Answer the user's question based on the provided context.

                Context:
                {context_text}

                Question: {request.query}

                Instructions:
                - Answer based on the context provided
                - If the context doesn't contain the answer, say so
                - Provide source attribution when possible
                - Be concise and accurate
                - Never hallucinate information not in the context
                """
            else:
                # No context available
                prompt = f"""
                You are an AI assistant for the AI Robotics textbook.
                The system couldn't find relevant context for the question: {request.query}
                Provide a general response based on your knowledge while noting that specific textbook content wasn't available.
                """

            # Call the OpenRouter API
            response = self.client.chat.completions.create(
                model=self.config.OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": self.config.AGENT_INSTRUCTIONS},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.MAX_RESPONSE_TOKENS,
                temperature=0.3  # Lower temperature for more consistent, factual responses
            )

            # Extract the response text
            ai_response = response.choices[0].message.content

            # Calculate confidence score based on retrieval
            confidence_score = 0.0
            if retrieved_chunks:
                avg_similarity = sum(chunk.similarity_score for chunk in retrieved_chunks) / len(retrieved_chunks)
                confidence_score = min(1.0, avg_similarity)  # Cap at 1.0
            else:
                confidence_score = 0.3  # Lower confidence when no context retrieved

            # Create the response object
            agent_response = AgentResponse(
                response=ai_response,
                query=request.query,
                retrieved_chunks=retrieved_chunks,
                source_attribution=list(source_attribution),
                confidence_score=confidence_score,
                query_type=request.query_type,
                conversation_id=conversation_id,
                has_sufficient_context=len(retrieved_chunks) > 0 or request.query_type == QueryType.USER_CONTEXT,
                hallucination_prevention_applied=True  # Always apply prevention
            )

            # Validate the response
            validator = ResponseValidator()
            validation_result = validator.validate_agent_response(agent_response)

            if not validation_result["is_valid"]:
                self.logger.warning(f"Response validation issues: {validation_result['issues']}")

            self.logger.info(f"Request processed successfully. Confidence: {confidence_score:.2f}, Chunks: {len(retrieved_chunks)}")
            return agent_response

        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            raise

    def validate_retrieval_connection(self) -> bool:
        """
        Validate that the retrieval service is accessible
        """
        try:
            return self.retrieval_tool.validate_retrieval_connection()
        except Exception as e:
            self.logger.error(f"Retrieval connection validation failed: {str(e)}")
            return False