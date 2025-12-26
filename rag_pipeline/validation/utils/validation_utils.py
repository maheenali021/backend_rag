"""
Utility functions for the retrieval validation system
"""
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from dataclasses import dataclass

from ..models.validation_models import RetrievalResult, ValidationRequest, ValidationResult, RetrievalQualityMetrics


@dataclass
class LogEntry:
    """
    Data class for structured logging of retrieval operations
    """
    timestamp: datetime
    operation: str
    query: str
    filters: Optional[Dict[str, Any]]
    results_count: int
    latency_ms: float
    source: str
    metadata: Optional[Dict[str, Any]] = None


class ValidationLogger:
    """
    Logger for retrieval validation operations with structured output
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def log_retrieval_operation(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        results: List[RetrievalResult],
        latency_ms: float,
        source: str = "retrieval_service"
    ):
        """
        Log a retrieval operation with structured data
        """
        log_entry = LogEntry(
            timestamp=datetime.now(),
            operation="retrieval",
            query=query,
            filters=filters,
            results_count=len(results),
            latency_ms=latency_ms,
            source=source
        )

        self.logger.info(
            f"RETRIEVAL_OPERATION - Query: '{query[:50]}...', "
            f"Results: {len(results)}, "
            f"Latency: {latency_ms:.2f}ms, "
            f"Filters: {filters}"
        )

    def log_validation_operation(
        self,
        validation_request: ValidationRequest,
        results: List[RetrievalResult],
        latency_ms: float,
        semantic_alignment: float,
        precision: float
    ):
        """
        Log a validation operation with structured data
        """
        self.logger.info(
            f"VALIDATION_OPERATION - Query: '{validation_request.query[:50]}...', "
            f"Results: {len(results)}, "
            f"Latency: {latency_ms:.2f}ms, "
            f"Semantic Alignment: {semantic_alignment:.3f}, "
            f"Precision: {precision:.3f}, "
            f"Filters: {validation_request.filters}"
        )

    def log_error(self, operation: str, query: str, error: str, source: str = "validation_service"):
        """
        Log an error during validation operations
        """
        self.logger.error(
            f"VALIDATION_ERROR - Operation: {operation}, "
            f"Query: '{query[:50]}...', "
            f"Source: {source}, "
            f"Error: {error}"
        )


class QualityMetricsCalculator:
    """
    Calculator for various quality metrics used in retrieval validation
    """
    @staticmethod
    def calculate_precision_at_k(
        retrieved_results: List[RetrievalResult],
        relevant_results: List[str],  # List of relevant document IDs or URLs
        k: int
    ) -> float:
        """
        Calculate precision at k (P@K) - precision considering only top-k results
        """
        if k <= 0 or not retrieved_results:
            return 0.0

        # Take only top-k results
        top_k_results = retrieved_results[:k]

        # Count how many of the top-k results are relevant
        relevant_retrieved = 0
        for result in top_k_results:
            if result.id in relevant_results or result.source_url in relevant_results:
                relevant_retrieved += 1

        return relevant_retrieved / min(k, len(top_k_results))

    @staticmethod
    def calculate_recall_at_k(
        retrieved_results: List[RetrievalResult],
        relevant_results: List[str],  # List of all relevant document IDs or URLs
        k: int
    ) -> float:
        """
        Calculate recall at k (R@K) - recall considering only top-k results
        """
        if not relevant_results:
            return 1.0  # If there are no relevant results, recall is 100%

        if k <= 0 or not retrieved_results:
            return 0.0

        # Take only top-k results
        top_k_results = retrieved_results[:k]

        # Count how many relevant results were retrieved in top-k
        relevant_retrieved = 0
        for result in top_k_results:
            if result.id in relevant_results or result.source_url in relevant_results:
                relevant_retrieved += 1

        return relevant_retrieved / len(relevant_results)

    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall
        """
        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def calculate_mean_reciprocal_rank(
        retrieved_results: List[RetrievalResult],
        relevant_results: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        """
        if not relevant_results or not retrieved_results:
            return 0.0

        for idx, result in enumerate(retrieved_results):
            if result.id in relevant_results or result.source_url in relevant_results:
                # Return reciprocal of the rank (1-indexed)
                return 1.0 / (idx + 1)

        # If no relevant results found, MRR is 0
        return 0.0

    @staticmethod
    def calculate_dcg(relevances: List[float]) -> float:
        """
        Calculate Discounted Cumulative Gain
        relevances: List of relevance scores (0.0 to 1.0) for retrieved documents
        """
        if not relevances:
            return 0.0

        dcg = relevances[0]  # First element has no discount
        for i in range(1, len(relevances)):
            # DCG formula: rel1 + sum(rel_i / log2(i+1))
            dcg += relevances[i] / (i + 1).bit_length() - 1  # This calculates log2(i+1)

        return dcg

    @staticmethod
    def calculate_ndcg(
        retrieved_results: List[RetrievalResult],
        ideal_relevances: List[float]
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain
        """
        if not retrieved_results or not ideal_relevances:
            return 0.0

        # Get relevance scores for retrieved results
        retrieved_relevances = [result.confidence_score for result in retrieved_results]

        # Calculate DCG for retrieved results
        dcg = QualityMetricsCalculator.calculate_dcg(retrieved_relevances)

        # Calculate ideal DCG (sorted by relevance)
        sorted_ideal = sorted(ideal_relevances, reverse=True)
        ideal_dcg = QualityMetricsCalculator.calculate_dcg(sorted_ideal)

        if ideal_dcg == 0:
            return 0.0

        return dcg / ideal_dcg


class PerformanceMonitor:
    """
    Monitor and track performance metrics for the retrieval system
    """
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.latency_samples = []
        self.start_time = datetime.now()

    def record_request(self, latency_ms: float, success: bool = True):
        """
        Record a request for performance tracking
        """
        self.request_count += 1
        if not success:
            self.error_count += 1

        self.latency_samples.append(latency_ms)

        # Keep only the last 1000 samples to avoid memory issues
        if len(self.latency_samples) > 1000:
            self.latency_samples = self.latency_samples[-1000:]

    def get_latency_percentile(self, percentile: float) -> float:
        """
        Get a specific latency percentile
        """
        if not self.latency_samples:
            return 0.0

        sorted_latencies = sorted(self.latency_samples)
        index = int(len(sorted_latencies) * (percentile / 100))
        index = min(index, len(sorted_latencies) - 1)  # Ensure we don't go out of bounds
        return sorted_latencies[index]

    def get_quality_metrics(self) -> RetrievalQualityMetrics:
        """
        Get comprehensive quality metrics
        """
        success_rate = 1.0 - (self.error_count / self.request_count if self.request_count > 0 else 0)

        return RetrievalQualityMetrics(
            precision=0.85,  # Placeholder - would be calculated from actual validation results
            recall=0.80,     # Placeholder - would be calculated from actual validation results
            f1_score=0.82,   # Placeholder - would be calculated from actual validation results
            semantic_alignment=0.88,  # Placeholder - would be calculated from actual validation results
            latency_p50=self.get_latency_percentile(50),
            latency_p95=self.get_latency_percentile(95),
            latency_p99=self.get_latency_percentile(99),
            success_rate=success_rate
        )

    def get_uptime_minutes(self) -> float:
        """
        Get the uptime in minutes since the monitor was started
        """
        uptime = datetime.now() - self.start_time
        return uptime.total_seconds() / 60


class ValidationUtils:
    """
    Main utility class combining all validation utilities
    """
    def __init__(self):
        self.logger = ValidationLogger()
        self.metrics_calculator = QualityMetricsCalculator()
        self.performance_monitor = PerformanceMonitor()

    def log_retrieval_with_results(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        results: List[RetrievalResult],
        latency_ms: float
    ):
        """
        Log retrieval operation with all relevant information
        """
        self.logger.log_retrieval_operation(query, filters, results, latency_ms)

    def calculate_validation_result(
        self,
        query: str,
        expected_sources: Optional[List[str]],
        retrieved_results: List[RetrievalResult],
        top_k: int = 5
    ) -> ValidationResult:
        """
        Calculate a complete validation result with all metrics
        """
        # Calculate precision at k
        precision_at_k = self.metrics_calculator.calculate_precision_at_k(
            retrieved_results, expected_sources or [], top_k
        )

        # Calculate recall at k
        recall_at_k = self.metrics_calculator.calculate_recall_at_k(
            retrieved_results, expected_sources or [], top_k
        )

        # Calculate F1 score
        f1_score = self.metrics_calculator.calculate_f1_score(precision_at_k, recall_at_k)

        # Create quality metrics object
        quality_metrics = RetrievalQualityMetrics(
            precision=precision_at_k,
            recall=recall_at_k,
            f1_score=f1_score,
            semantic_alignment=0.85,  # This would be calculated separately
            latency_p50=0.0,  # Would be filled in by caller
            latency_p95=0.0,  # Would be filled in by caller
            latency_p99=0.0,  # Would be filled in by caller
            success_rate=1.0  # Would be calculated separately
        )

        # Determine if validation passes based on thresholds
        is_valid = (
            precision_at_k >= 0.70 and  # At least 70% precision
            recall_at_k >= 0.50 and    # At least 50% recall
            f1_score >= 0.60           # At least 60% F1 score
        )

        # Get retrieved source URLs
        retrieved_sources = [result.source_url for result in retrieved_results if result.source_url]

        return ValidationResult(
            query=query,
            expected_sources=expected_sources,
            retrieved_sources=retrieved_sources,
            quality_metrics=quality_metrics,
            is_valid=is_valid,
            details={
                "top_k": top_k,
                "retrieved_count": len(retrieved_results)
            }
        )


# Global instance of validation utilities
validation_utils = ValidationUtils()