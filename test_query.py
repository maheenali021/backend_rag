#!/usr/bin/env python3
"""
Test script to verify the RAG agent can handle user queries
"""
import sys
import os

# Add the backend directory to the path so we can import from rag_agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_agent.services.rag_agent_service import RAGAgentService
from rag_agent.models.agent_models import AgentRequest, QueryType

def test_query():
    print("Testing RAG Agent with query: 'What is this book about?'")
    print("="*60)

    try:
        # Create the agent service
        print("Creating RAG Agent Service...")
        agent_service = RAGAgentService()
        print("[SUCCESS] RAG Agent Service created successfully")

        # Create a test request
        request = AgentRequest(
            query="What is this book about?",
            query_type=QueryType.GENERAL,
            top_k=5
        )
        print(f"[SUCCESS] Request created: {request.query}")

        # Process the request
        print("\nProcessing request...")
        response = agent_service.process_request(request)

        print("[SUCCESS] Request processed successfully!")
        print(f"\nResponse: {response.response}")
        print(f"Confidence Score: {response.confidence_score:.2f}")
        print(f"Retrieved Chunks: {len(response.retrieved_chunks)}")
        print(f"Source Attribution: {len(response.source_attribution)} sources")

        if response.retrieved_chunks:
            print(f"\nTop retrieved chunk preview:")
            first_chunk = response.retrieved_chunks[0]
            print(f"  Content preview: {first_chunk.content[:200]}...")
            print(f"  Similarity Score: {first_chunk.similarity_score:.2f}")
            print(f"  Source: {first_chunk.source_url}")

        return response

    except Exception as e:
        print(f"[ERROR] Error during query test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_query()