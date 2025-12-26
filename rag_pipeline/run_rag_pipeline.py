"""
Main script to run the complete RAG pipeline
This script orchestrates the entire process from content extraction to vector storage
"""
import sys
import os
import argparse
from typing import Dict, Any

# Add the parent directory to the path so we can import from rag_pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_pipeline.pipelines.rag_pipeline import RAGPipeline
from rag_pipeline.config import Config
from rag_pipeline.utils.logging_config import get_logger


def main():
    """
    Main function to run the RAG pipeline
    """
    parser = argparse.ArgumentParser(description='Run the RAG Pipeline')
    parser.add_argument('--url', type=str, help='Website URL to process (optional, defaults to config)')
    parser.add_argument('--validate', action='store_true', help='Perform validation steps (default: True)')
    parser.add_argument('--incremental', action='store_true', help='Run incremental pipeline instead of full pipeline')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')

    args = parser.parse_args()

    # Validate configuration
    try:
        Config.validate()
        print("Configuration validation passed")
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)

    # Initialize logger
    logger = get_logger()
    logger.info("Starting RAG Pipeline execution")

    # Initialize the RAG pipeline
    rag_pipeline = RAGPipeline()

    try:
        if args.incremental:
            print("Running incremental RAG pipeline...")
            logger.info("Executing incremental pipeline")
            results = rag_pipeline.run_incremental_pipeline(args.url)
        else:
            print("Running complete RAG pipeline...")
            logger.info("Executing complete pipeline")
            results = rag_pipeline.run_complete_pipeline(args.url, validate=args.validate)

        # Print results
        print("\n" + "="*50)
        print("RAG PIPELINE RESULTS")
        print("="*50)
        print(f"Status: {results.get('status', 'unknown')}")
        print(f"Content chunks processed: {results.get('content_chunks_count', 0)}")
        print(f"Embeddings generated: {results.get('embeddings_count', 0)}")
        print(f"Items stored: {results.get('storage_count', 0)}")

        validation = results.get('validation', {})
        if validation:
            print(f"Validation passed: {validation.get('valid', False)}")

        metrics = results.get('metrics', {})
        if metrics:
            content_metrics = metrics.get('content', {})
            embedding_metrics = metrics.get('embeddings', {})

            print(f"Average chunk size: {content_metrics.get('average_chunk_size', 0):.2f} chars")
            print(f"Generation success rate: {embedding_metrics.get('generation_success_rate', 0):.2%}")

        print("="*50)

        # Check if the pipeline met success criteria
        success = _check_success_criteria(results)
        if success:
            print("SUCCESS: All success criteria met!")
            return 0
        else:
            print("FAILURE: Some success criteria not met")
            return 1

    except Exception as e:
        logger.error(f"Error running RAG pipeline: {str(e)}")
        print(f"Error running RAG pipeline: {str(e)}")
        return 1


def _check_success_criteria(results: Dict[str, Any]) -> bool:
    """
    Check if the pipeline results meet the success criteria defined in the specification

    Success criteria from spec:
    - SC-004: Embedding generation achieves 95% success rate across all content chunks
    - SC-005: Embeddings and metadata are successfully stored in Qdrant Cloud with 99% reliability
    - SC-006: Semantic queries return relevant results within 2 seconds with 90% accuracy (partially testable)
    """
    try:
        content_count = results.get('content_chunks_count', 0)
        embedding_count = results.get('embeddings_count', 0)
        storage_count = results.get('storage_count', 0)

        # Calculate success rates
        embedding_success_rate = embedding_count / content_count if content_count > 0 else 0
        storage_reliability = storage_count / embedding_count if embedding_count > 0 else 0

        # Check individual criteria
        criterion_004_met = embedding_success_rate >= 0.95  # 95% embedding success rate
        criterion_005_met = storage_reliability >= 0.99    # 99% storage reliability

        print(f"\nSuccess Criteria Check:")
        print(f"  SC-004 (95% embedding success): {'SUCCESS' if criterion_004_met else 'FAILURE'} "
              f"({embedding_success_rate:.2%})")
        print(f"  SC-005 (99% storage reliability): {'SUCCESS' if criterion_005_met else 'FAILURE'} "
              f"({storage_reliability:.2%})")

        return criterion_004_met and criterion_005_met

    except Exception as e:
        print(f"Error checking success criteria: {e}")
        return False


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)