"""
Evaluation library for geo-experiments.

This module provides tools for evaluating the performance of different
assignment and reporting methods across multiple simulation runs.
"""

from .metrics import EvaluationRunner

__all__ = ['EvaluationRunner']