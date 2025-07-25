"""
Assignment methods for geo-experiments.

This module provides different strategies for assigning geos to 
treatment and control groups.
"""

from .methods import RandomAssignment, KMeansEmbeddingAssignment, PrognosticScoreAssignment, EmbeddingBasedAssignment, HybridEmbeddingAssignment
from .spatial_utils import add_spectral_spatial_embedding
from .stratified_utils import stratified_assignment_within_clusters, evaluate_cluster_balance, print_balance_summary

__all__ = [
    'RandomAssignment', 
    'KMeansEmbeddingAssignment', 
    'PrognosticScoreAssignment',
    'EmbeddingBasedAssignment',
    'HybridEmbeddingAssignment',
    'add_spectral_spatial_embedding',
    'stratified_assignment_within_clusters',
    'evaluate_cluster_balance',
    'print_balance_summary'
]