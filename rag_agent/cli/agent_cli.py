#!/usr/bin/env python3
"""
CLI script for testing the RAG Agent
Provides command-line interface for testing agent functionality
"""
import argparse
import sys
import os
import json
from typing import Dict, Any
import asyncio

# Add the backend directory to the path so we can import from rag_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent.models.agent_models import AgentRequest, QueryType
from rag_agent.services.rag_agent_service import RAGAgentService
from rag_agent.config import Config


def create_test_request(query: str, query_type: QueryType = QueryType.GENERAL,
                       chapter_filter: str = None, user_context: str = None) -> AgentRequest:
    """Create a test agent request"""
    return AgentRequest(
        query=query,
        query_type=query_type,
        chapter_filter=chapter_filter,
        user_context=user_context,
        top_k=5
    )


def print_response(response):
    """Pretty print the agent response"""
    print("\n" + "="*60)
    print("AGENT RESPONSE")
    print("="*60)
    print(f"Query: {response.query}")
    print(f"Response: {response.response}")
    print(f"Confidence: {response.confidence_score:.2f}")
    print(f"Query Type: {response.query_type}")
    print(f"Sufficient Context: {response.has_sufficient_context}")
    print(f"Hallucination Prevention Applied: {response.hallucination_prevention_applied}")

    if response.source_attribution:
        print(f"Sources: {', '.join(response.source_attribution[:3])}")  # Show first 3 sources

    if response.retrieved_chunks:
        print(f"Retrieved Chunks: {len(response.retrieved_chunks)}")
        for i, chunk in enumerate(response.retrieved_chunks[:3], 1):  # Show first 3 chunks
            print(f"  {i}. Chapter: {chunk.chapter or 'N/A'}, Score: {chunk.similarity_score:.2f}")
            print(f"     Preview: {chunk.content[:100]}...")

    print("="*60 + "\n")


def interactive_mode():
    """Run the agent in interactive mode"""
    print("RAG Agent Interactive Mode")
    print("Type 'quit' or 'exit' to exit")
    print("Type 'help' for commands")

    agent_service = RAGAgentService()

    while True:
        try:
            user_input = input("\nYour question: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if user_input.lower() == 'help':
                print("\nCommands:")
                print("  - Type any question to get an answer from the agent")
                print("  - Use 'chapter:<chapter_name> <question>' for chapter-specific queries")
                print("  - Use 'context:<your_context> <question>' for user context mode")
                print("  - Type 'quit' or 'exit' to exit")
                continue

            # Parse special commands
            query_type = QueryType.GENERAL
            chapter_filter = None
            user_context = None

            if user_input.lower().startswith('chapter:'):
                parts = user_input.split(' ', 1)
                if len(parts) > 1:
                    chapter_part = parts[0][8:]  # Remove 'chapter:' prefix
                    user_input = parts[1]
                    query_type = QueryType.CHAPTER_SPECIFIC
                    chapter_filter = chapter_part
            elif user_input.lower().startswith('context:'):
                parts = user_input.split(' ', 1)
                if len(parts) > 1:
                    context_part = parts[0][8:]  # Remove 'context:' prefix
                    user_input = parts[1]
                    query_type = QueryType.USER_CONTEXT
                    user_context = context_part

            # Create and process request
            request = create_test_request(
                query=user_input,
                query_type=query_type,
                chapter_filter=chapter_filter,
                user_context=user_context
            )

            response = agent_service.process_request(request)
            print_response(response)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def run_single_query(query: str, query_type: str = "general", chapter: str = None, context: str = None):
    """Run a single query and display the result"""
    agent_service = RAGAgentService()

    # Convert query type string to enum
    qt = QueryType.GENERAL
    if query_type == "chapter_specific":
        qt = QueryType.CHAPTER_SPECIFIC
    elif query_type == "user_context":
        qt = QueryType.USER_CONTEXT

    request = create_test_request(
        query=query,
        query_type=qt,
        chapter_filter=chapter,
        user_context=context
    )

    response = agent_service.process_request(request)
    print_response(response)


def batch_test(queries_file: str):
    """Run batch tests from a file"""
    agent_service = RAGAgentService()

    with open(queries_file, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)

    print(f"Running batch test with {len(queries_data)} queries...")

    results = []
    for i, query_item in enumerate(queries_data):
        print(f"Processing query {i+1}/{len(queries_data)}: {query_item['query'][:50]}...")

        request = create_test_request(
            query=query_item['query'],
            query_type=QueryType(query_item.get('query_type', 'general')),
            chapter_filter=query_item.get('chapter_filter'),
            user_context=query_item.get('user_context')
        )

        response = agent_service.process_request(request)
        results.append({
            "query": query_item['query'],
            "response": response.response,
            "confidence": response.confidence_score,
            "retrieved_chunks": len(response.retrieved_chunks)
        })

    # Print summary
    print(f"\nBatch test completed! Processed {len(results)} queries.")
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    print(f"Average confidence: {avg_confidence:.2f}")

    # Optionally save results
    output_file = f"batch_results_{len(results)}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='RAG Agent CLI')
    parser.add_argument('query', nargs='?', help='Query to process (interactive mode if not provided)')
    parser.add_argument('--type', choices=['general', 'chapter_specific', 'user_context'],
                       default='general', help='Type of query')
    parser.add_argument('--chapter', help='Chapter filter for chapter-specific queries')
    parser.add_argument('--context', help='User-provided context')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--batch', help='Run batch test from JSON file')
    parser.add_argument('--validate-config', action='store_true', help='Validate configuration')

    args = parser.parse_args()

    if args.validate_config:
        try:
            Config.validate()
            print("SUCCESS: Configuration is valid")
            return 0
        except ValueError as e:
            print(f"FAILURE: Configuration error: {e}")
            return 1

    if args.batch:
        batch_test(args.batch)
        return 0

    if args.interactive or not args.query:
        interactive_mode()
        return 0

    # Run single query
    run_single_query(args.query, args.type, args.chapter, args.context)
    return 0


if __name__ == "__main__":
    sys.exit(main())