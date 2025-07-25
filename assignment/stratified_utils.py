"""
Stratified assignment utilities for geo-experiments.

This module provides functions for translating clusters into balanced 
treatment/control assignments using stratified sampling.
"""

import numpy as np
import pandas as pd
from typing import Optional


def stratified_assignment_within_clusters(
    geo_features: pd.DataFrame,
    cluster_labels: np.ndarray,
    treatment_ratio: float = 0.5,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Perform stratified assignment within clusters.
    
    This function takes cluster assignments and creates treatment/control
    assignments by randomly sampling within each cluster. This ensures
    that each cluster contributes proportionally to both treatment and
    control groups, leading to better balance.
    
    Args:
        geo_features: DataFrame with geo-level features (must have 'geo' column)
        cluster_labels: Array of cluster assignments for each geo
        treatment_ratio: Proportion to assign to treatment within each cluster
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns ['geo', 'assignment', 'cluster'] where assignment
        is either 'treatment' or 'control'
        
    Raises:
        ValueError: If inputs are malformed
    """
    if 'geo' not in geo_features.columns:
        raise ValueError("geo_features must contain 'geo' column")
    
    if len(geo_features) != len(cluster_labels):
        raise ValueError("geo_features and cluster_labels must have same length")
    
    if not 0 < treatment_ratio < 1:
        raise ValueError("treatment_ratio must be between 0 and 1")
    
    if seed is not None:
        np.random.seed(seed)
    
    assignments = []
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        # Get geos in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_geos = geo_features.loc[cluster_mask, 'geo'].tolist()
        n_cluster_geos = len(cluster_geos)
        
        if n_cluster_geos == 0:
            continue
        
        # Calculate number for treatment in this cluster
        n_treatment = int(n_cluster_geos * treatment_ratio)
        
        # Randomly select treatment geos within this cluster
        treatment_geos = np.random.choice(
            cluster_geos, 
            size=n_treatment, 
            replace=False
        )
        
        # Create assignments for this cluster
        for geo in cluster_geos:
            assignment = 'treatment' if geo in treatment_geos else 'control'
            assignments.append({
                'geo': geo, 
                'assignment': assignment,
                'cluster': cluster_id
            })
    
    return pd.DataFrame(assignments)


def evaluate_cluster_balance(
    geo_features: pd.DataFrame,
    assignment_df: pd.DataFrame,
    feature_cols: list
) -> pd.DataFrame:
    """
    Evaluate balance of features within and across clusters.
    
    Args:
        geo_features: DataFrame with geo-level features
        assignment_df: DataFrame with assignment results (should have 'cluster' column)
        feature_cols: List of feature columns to evaluate balance for
        
    Returns:
        DataFrame with balance statistics by cluster and overall
    """
    # Handle case where assignment_df doesn't have cluster column (e.g., RandomAssignment)
    assignment_df_copy = assignment_df.copy()
    if 'cluster' not in assignment_df_copy.columns:
        assignment_df_copy['cluster'] = 0  # Put all geos in one cluster
    
    # Merge assignment with features
    merged = geo_features.merge(assignment_df_copy, on='geo')
    
    balance_stats = []
    
    # Overall balance (ignoring clusters)
    for feature in feature_cols:
        treat_mean = merged[merged['assignment'] == 'treatment'][feature].mean()
        control_mean = merged[merged['assignment'] == 'control'][feature].mean()
        pooled_std = merged[feature].std()
        
        # Standardized mean difference
        smd = abs(treat_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        balance_stats.append({
            'cluster': 'Overall',
            'feature': feature,
            'treatment_mean': treat_mean,
            'control_mean': control_mean,
            'difference': treat_mean - control_mean,
            'standardized_diff': smd
        })
    
    # Balance within each cluster
    for cluster_id in merged['cluster'].unique():
        cluster_data = merged[merged['cluster'] == cluster_id]
        
        for feature in feature_cols:
            treat_data = cluster_data[cluster_data['assignment'] == 'treatment'][feature]
            control_data = cluster_data[cluster_data['assignment'] == 'control'][feature]
            
            if len(treat_data) > 0 and len(control_data) > 0:
                treat_mean = treat_data.mean()
                control_mean = control_data.mean()
                pooled_std = cluster_data[feature].std()
                
                smd = abs(treat_mean - control_mean) / pooled_std if pooled_std > 0 else 0
                
                balance_stats.append({
                    'cluster': f'Cluster_{cluster_id}',
                    'feature': feature,
                    'treatment_mean': treat_mean,
                    'control_mean': control_mean,
                    'difference': treat_mean - control_mean,
                    'standardized_diff': smd
                })
    
    return pd.DataFrame(balance_stats)


def print_balance_summary(balance_df: pd.DataFrame, threshold: float = 0.1):
    """
    Print a summary of balance statistics.
    
    Args:
        balance_df: DataFrame from evaluate_cluster_balance()
        threshold: Threshold for concerning imbalance (standardized mean difference)
    """
    print("🔍 Balance Summary:")
    print("=" * 50)
    
    # Overall balance
    overall_balance = balance_df[balance_df['cluster'] == 'Overall']
    print("\n📊 Overall Balance (across all geos):")
    for _, row in overall_balance.iterrows():
        status = "✅" if abs(row['standardized_diff']) < threshold else "⚠️"
        print(f"  {status} {row['feature']}: SMD = {row['standardized_diff']:.3f}")
    
    # Cluster-level balance
    cluster_balance = balance_df[balance_df['cluster'] != 'Overall']
    if len(cluster_balance) > 0:
        print(f"\n🎯 Within-Cluster Balance:")
        
        # Average balance across clusters
        avg_smd = cluster_balance.groupby('feature')['standardized_diff'].mean()
        max_smd = cluster_balance.groupby('feature')['standardized_diff'].max()
        
        for feature in avg_smd.index:
            avg_val = avg_smd[feature]
            max_val = max_smd[feature]
            status = "✅" if max_val < threshold else "⚠️"
            print(f"  {status} {feature}: Avg SMD = {avg_val:.3f}, Max SMD = {max_val:.3f}")
    
    print(f"\n📝 Note: SMD < {threshold} indicates good balance")
    print("SMD = Standardized Mean Difference between treatment and control")