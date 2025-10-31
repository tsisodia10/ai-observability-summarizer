"""
Centralized question classification and pattern matching for observability services.

This module provides generic question classification that can be used across
different observability services (Tempo, Prometheus, etc.).
"""

import re
from typing import Dict, Any, List, Optional
from enum import Enum


class QuestionType(Enum):
    """Enumeration of different question types for better query classification."""
    ERROR_TRACES = "error_traces"
    SLOW_TRACES = "slow_traces"
    FAST_TRACES = "fast_traces"
    SERVICE_ACTIVITY = "service_activity"
    DETAILED_ANALYSIS = "detailed_analysis"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    ALERT_ANALYSIS = "alert_analysis"
    METRIC_ANALYSIS = "metric_analysis"
    GENERAL = "general"


class QuestionClassifier:
    """Generic question classifier for observability services."""

    # Define question patterns with their corresponding question types
    QUESTION_PATTERNS = {
        QuestionType.ERROR_TRACES: [
            r"error", r"failed", r"exception", r"fault", r"problem",
            r"issue", r"broken", r"malfunction", r"defect", r"crash"
        ],
        QuestionType.SLOW_TRACES: [
            r"slow", r"slowest", r"latency", r"delay", r"performance",
            r"bottleneck", r"lag", r"timeout", r"response time"
        ],
        QuestionType.FAST_TRACES: [
            r"fast", r"fastest", r"quick", r"rapid", r"efficient"
        ],
        QuestionType.SERVICE_ACTIVITY: [
            r"active", r"busy", r"services", r"service activity",
            r"most active", r"running", r"operational", r"traffic"
        ],
        QuestionType.DETAILED_ANALYSIS: [
            r"top", r"slowest", r"fastest", r"request flow",
            r"detailed analysis", r"analyze", r"analysis", r"breakdown"
        ],
        QuestionType.PERFORMANCE_ANALYSIS: [
            r"performance", r"throughput", r"capacity", r"utilization",
            r"resource usage", r"efficiency", r"optimization"
        ],
        QuestionType.ALERT_ANALYSIS: [
            r"alert", r"alerts", r"warning", r"critical", r"firing",
            r"notifications", r"incidents"
        ],
        QuestionType.METRIC_ANALYSIS: [
            r"metric", r"metrics", r"measurement", r"statistics",
            r"data", r"values", r"trends"
        ]
    }

    @classmethod
    def get_all_patterns(cls) -> Dict[QuestionType, List[str]]:
        """
        Get all question patterns.

        Returns:
            Dict mapping question types to their patterns
        """
        return cls.QUESTION_PATTERNS.copy()

    @classmethod
    def classify_question(cls, question: str) -> QuestionType:
        """
        Classify a user question to determine the appropriate query type.

        Args:
            question: The user's question

        Returns:
            QuestionType: The classified question type
        """
        question_lower = question.lower()

        for question_type, patterns in cls.QUESTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    return question_type

        return QuestionType.GENERAL

    @classmethod
    def get_confidence_score(cls, question: str, question_type: QuestionType) -> float:
        """
        Calculate confidence score for a classification.

        Args:
            question: The user's question
            question_type: The classified question type

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        question_lower = question.lower()
        patterns = cls.QUESTION_PATTERNS.get(question_type, [])
        
        if not patterns:
            return 0.0
        
        matches = 0
        for pattern in patterns:
            if re.search(pattern, question_lower, re.IGNORECASE):
                matches += 1
        
        return min(matches / len(patterns), 1.0)

    @classmethod
    def extract_key_concepts(cls, question: str) -> Dict[str, Any]:
        """
        Extract key concepts from a question.

        Args:
            question: The user's question

        Returns:
            Dict containing extracted concepts
        """
        question_lower = question.lower()
        concepts = {
            "time_indicators": [],
            "service_indicators": [],
            "metric_indicators": [],
            "action_indicators": []
        }

        # Time indicators
        time_patterns = [
            r"last\s+(\d+)\s+(hour|hours|day|days|week|weeks)",
            r"past\s+(\d+)\s+(hour|hours|day|days|week|weeks)",
            r"(\d+)\s+(hour|hours|day|days|week|weeks)\s+ago",
            r"today", r"yesterday", r"this\s+week", r"this\s+month"
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, question_lower)
            concepts["time_indicators"].extend(matches)

        # Service indicators
        service_patterns = [
            r"service[s]?\s+(\w+)",
            r"(\w+)\s+service",
            r"pod[s]?\s+(\w+)",
            r"deployment[s]?\s+(\w+)"
        ]
        
        for pattern in service_patterns:
            matches = re.findall(pattern, question_lower)
            concepts["service_indicators"].extend(matches)

        # Metric indicators
        metric_patterns = [
            r"cpu", r"memory", r"disk", r"network", r"latency",
            r"throughput", r"requests", r"response\s+time"
        ]
        
        for pattern in metric_patterns:
            if re.search(pattern, question_lower):
                concepts["metric_indicators"].append(pattern)

        # Action indicators
        action_patterns = [
            r"show", r"list", r"find", r"get", r"analyze",
            r"compare", r"trend", r"summary"
        ]
        
        for pattern in action_patterns:
            if re.search(pattern, question_lower):
                concepts["action_indicators"].append(pattern)

        return concepts


class TempoQuestionClassifier(QuestionClassifier):
    """Tempo-specific question classifier with TraceQL query generation."""

    # TraceQL query constants
    QUERY_ERROR_TRACES = "status=error"
    QUERY_SLOW_TRACES = "duration>1s"
    QUERY_ALL_SERVICES = "service.name=*"

    @classmethod
    def get_trace_query(cls, question_type: QuestionType, question: str) -> str:
        """
        Get the appropriate TraceQL query based on the question type.

        Args:
            question_type: The classified question type
            question: The original question (for additional context)

        Returns:
            str: The appropriate TraceQL query
        """
        question_lower = question.lower()

        if question_type == QuestionType.ERROR_TRACES:
            return cls.QUERY_ERROR_TRACES
        elif question_type == QuestionType.SLOW_TRACES and "fastest" not in question_lower:
            return cls.QUERY_SLOW_TRACES
        else:
            # Default query for FAST_TRACES, SERVICE_ACTIVITY, DETAILED_ANALYSIS, and GENERAL
            return cls.QUERY_ALL_SERVICES


class PrometheusQuestionClassifier(QuestionClassifier):
    """Prometheus-specific question classifier with PromQL query generation."""

    @classmethod
    def get_promql_query(cls, question_type: QuestionType, question: str, 
                        metric_name: Optional[str] = None) -> str:
        """
        Get the appropriate PromQL query based on the question type.

        Args:
            question_type: The classified question type
            question: The original question
            metric_name: Optional specific metric name

        Returns:
            str: The appropriate PromQL query
        """
        if not metric_name:
            return "up"  # Default query
        
        if question_type == QuestionType.PERFORMANCE_ANALYSIS:
            return f"rate({metric_name}[5m])"
        elif question_type == QuestionType.ALERT_ANALYSIS:
            return "ALERTS"
        else:
            return metric_name


class TraceErrorDetector:
    """Detects errors in trace data using pattern matching."""

    # Error indicators to look for in traces
    ERROR_PATTERNS = [
        r"\berror\b",
        r"\bfailed\b",
        r"\bexception\b",
        r"\bfault\b",
        r"\bstatus[:\s]*error\b",
        r"\bstatus[:\s]*failed\b",
        r"http[:\s]*[45]\d{2}",  # HTTP 4xx/5xx status codes
    ]

    @classmethod
    def is_error_trace(cls, trace: Dict[str, Any]) -> bool:
        """
        Check if a trace contains error indicators.

        Args:
            trace: Trace data dictionary

        Returns:
            bool: True if trace appears to contain errors
        """
        trace_str = str(trace).lower()

        for pattern in cls.ERROR_PATTERNS:
            if re.search(pattern, trace_str, re.IGNORECASE):
                return True

        return False

    @classmethod
    def extract_error_details(cls, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract detailed error information from a trace.

        Args:
            trace: Trace data dictionary

        Returns:
            Dict containing error details
        """
        trace_str = str(trace).lower()
        error_details = {
            "has_error": False,
            "error_type": None,
            "error_message": None,
            "status_code": None
        }

        for pattern in cls.ERROR_PATTERNS:
            match = re.search(pattern, trace_str, re.IGNORECASE)
            if match:
                error_details["has_error"] = True
                error_details["error_type"] = pattern
                error_details["error_message"] = match.group(0)
                
                # Extract HTTP status codes
                status_match = re.search(r"http[:\s]*(\d{3})", trace_str)
                if status_match:
                    error_details["status_code"] = int(status_match.group(1))
                break

        return error_details
