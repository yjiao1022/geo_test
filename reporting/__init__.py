"""
Reporting models for geo-experiments.

This module provides different models for generating counterfactual estimates
and computing treatment effects in geo-experiments.
"""

from .models import MeanMatchingModel

__all__ = ['MeanMatchingModel']