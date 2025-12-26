#!/usr/bin/env python3
"""
CLI script for batch validation of the retrieval pipeline
"""
import asyncio
import argparse
import json
import csv
import sys
from typing import List, Dict, Any
import time
import logging
from pathlib import Path

# Add the parent directory to the path so we can import from validation modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation.services.retrieval_service import retrieval_service
from validation.models.validation_models import ValidationRequest, ValidationResult
from validation.utils.validation_utils import validation_utils


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def validate_single_query(
    query: str,
    expected_sources: List[str] = None,
    filters: Dict[str, Any] = None,
    top_k: int = 5
) -> ValidationResult:
    """Validate a single query"""
    try:
        # Create validation request
        validation_request = ValidationRequest(
            query=query,
            expected_sources=expected_sources,
            filters=filters,
            top_k=top_k
        )

        # Perform retrieval
        start_time = time.time()
        results = await retrieval_service.retrieve_similar_chunks(validation_request)
        latency = (time.time() - start_time) * 1000

        # Calculate metrics
        semantic_alignment = await retrieval_service.calculate_semantic_alignment(query, results)
        precision = await retrieval_service.calculate_precision(results, expected_sources)

        # Create validation result
        validation_result = validation_utils.calculate_validation_result(
            query=query,
            expected_sources=expected_sources,
            retrieved_results=results,
            top_k=top_k
        )

        # Update with actual metrics
        validation_result.quality_metrics.semantic_alignment = semantic_alignment
        validation_result.quality_metrics.precision = precision
        validation_result.quality_metrics.latency_p50 = latency
        validation_result.quality_metrics.latency_p95 = latency
        validation_result.quality_metrics.latency_p99 = latency

        return validation_result

    except Exception as e:
        logging.error(f"Error validating query '{query}': {str(e)}")
        return ValidationResult(
            query=query,
            expected_sources=expected_sources,
            retrieved_sources=[],
            quality_metrics=None,
            is_valid=False,
            details={"error": str(e)}
        )


async def run_batch_validation_from_file(
    input_file: str,
    output_file: str,
    format: str = "json"
) -> List[ValidationResult]:
    """Run batch validation from an input file"""
    results = []

    # Determine input format based on file extension if not specified
    if not format:
        if input_file.endswith('.csv'):
            format = 'csv'
        elif input_file.endswith('.json'):
            format = 'json'
        else:
            format = 'txt'

    if format.lower() == 'csv':
        # Read queries from CSV file
        with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                query = row.get('query', '').strip()
                if query:
                    expected_sources_str = row.get('expected_sources', '')
                    expected_sources = expected_sources_str.split('|') if expected_sources_str else None

                    filters_str = row.get('filters', '')
                    filters = json.loads(filters_str) if filters_str else None

                    top_k = int(row.get('top_k', 5))

                    result = await validate_single_query(query, expected_sources, filters, top_k)
                    results.append(result)
    elif format.lower() == 'json':
        # Read queries from JSON file
        with open(input_file, 'r', encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)

            # Expecting either a list of queries or a list of objects with query details
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        # Simple list of queries
                        result = await validate_single_query(item)
                        results.append(result)
                    elif isinstance(item, dict):
                        # Object with query details
                        query = item.get('query', '').strip()
                        if query:
                            expected_sources = item.get('expected_sources')
                            filters = item.get('filters')
                            top_k = item.get('top_k', 5)

                            result = await validate_single_query(query, expected_sources, filters, top_k)
                            results.append(result)
    else:  # txt format - one query per line
        with open(input_file, 'r', encoding='utf-8') as txtfile:
            for line in txtfile:
                query = line.strip()
                if query and not query.startswith('#'):  # Skip empty lines and comments
                    result = await validate_single_query(query)
                    results.append(result)

    # Write results to output file
    if output_file:
        if output_file.endswith('.json'):
            with open(output_file, 'w', encoding='utf-8') as f:
                json_results = []
                for result in results:
                    json_results.append({
                        "query": result.query,
                        "expected_sources": result.expected_sources,
                        "retrieved_sources": result.retrieved_sources,
                        "quality_metrics": result.quality_metrics.dict() if result.quality_metrics else None,
                        "is_valid": result.is_valid,
                        "details": result.details
                    })
                json.dump(json_results, f, indent=2, default=str)
        elif output_file.endswith('.csv'):
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['query', 'is_valid', 'precision', 'recall', 'f1_score',
                             'semantic_alignment', 'retrieved_count', 'expected_count']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    writer.writerow({
                        'query': result.query,
                        'is_valid': result.is_valid,
                        'precision': result.quality_metrics.precision if result.quality_metrics else None,
                        'recall': result.quality_metrics.recall if result.quality_metrics else None,
                        'f1_score': result.quality_metrics.f1_score if result.quality_metrics else None,
                        'semantic_alignment': result.quality_metrics.semantic_alignment if result.quality_metrics else None,
                        'retrieved_count': len(result.retrieved_sources),
                        'expected_count': len(result.expected_sources) if result.expected_sources else 0
                    })

    return results


async def print_summary(results: List[ValidationResult]):
    """Print a summary of validation results"""
    if not results:
        print("No results to summarize.")
        return

    total_queries = len(results)
    valid_queries = sum(1 for r in results if r.is_valid)
    invalid_queries = total_queries - valid_queries

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries processed: {total_queries}")
    print(f"Valid queries: {valid_queries}")
    print(f"Invalid queries: {invalid_queries}")
    print(f"Success rate: {valid_queries/total_queries*100:.2f}%")

    # Calculate average metrics if available
    if results[0].quality_metrics:
        avg_precision = sum(r.quality_metrics.precision for r in results if r.quality_metrics) / len(results)
        avg_recall = sum(r.quality_metrics.recall for r in results if r.quality_metrics) / len(results)
        avg_f1 = sum(r.quality_metrics.f1_score for r in results if r.quality_metrics) / len(results)
        avg_alignment = sum(r.quality_metrics.semantic_alignment for r in results if r.quality_metrics) / len(results)

        print(f"\nAverage metrics:")
        print(f"  Precision: {avg_precision:.3f}")
        print(f"  Recall: {avg_recall:.3f}")
        print(f"  F1-Score: {avg_f1:.3f}")
        print(f"  Semantic Alignment: {avg_alignment:.3f}")

    print(f"{'='*60}")


async def main():
    parser = argparse.ArgumentParser(description='Batch validation of retrieval pipeline')
    parser.add_argument('input_file', help='Input file with queries to validate (CSV, JSON, or TXT)')
    parser.add_argument('-o', '--output', help='Output file to save results (JSON or CSV)')
    parser.add_argument('-f', '--format', choices=['csv', 'json', 'txt'], help='Input file format')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to retrieve (default: 5)')

    args = parser.parse_args()

    setup_logging(args.verbose)

    logging.info(f"Starting batch validation from {args.input_file}")
    logging.info(f"Output will be saved to {args.output or 'console'}")

    try:
        # Run batch validation
        results = await run_batch_validation_from_file(
            args.input_file,
            args.output,
            args.format
        )

        # Print summary
        await print_summary(results)

        # Exit with error code if there are validation failures
        invalid_count = sum(1 for r in results if not r.is_valid)
        if invalid_count > 0:
            logging.warning(f"{invalid_count} queries failed validation")
            sys.exit(1)
        else:
            logging.info("All queries passed validation!")
            sys.exit(0)

    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during batch validation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())