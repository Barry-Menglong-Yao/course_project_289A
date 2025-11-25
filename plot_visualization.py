#!/usr/bin/env python3
"""
Script to plot correlation between max_forgetting scores and graph metrics.
For each metric type, creates a scatter plot showing the correlation.
"""

import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def extract_metric_value(metrics, metric_name):
    """从metrics字典中提取指定指标的值"""
    # 特殊处理：直接数值
    if metric_name in ['density', 'transitivity', 'degree_assortativity', 'global_efficiency']:
        return metrics.get(metric_name)
    
    # 特殊处理：需要从子字典中提取mean
    if metric_name in ['betweenness_centrality', 'eigenvector_centrality', 'closeness_centrality', 'clustering']:
        metric_data = metrics.get(metric_name)
        if isinstance(metric_data, dict):
            return metric_data.get('mean')
        return None
    
    # 特殊处理：degree_statistics
    if metric_name in ['mean_degree', 'mean_in_degree', 'mean_out_degree']:
        degree_stats = metrics.get('degree_statistics', {})
        return degree_stats.get(metric_name)
    
    # 特殊处理：average_shortest_path_length, diameter
    if metric_name in ['average_shortest_path_length', 'diameter']:
        metric_data = metrics.get(metric_name)
        # If it's a dict with error, return None
        if isinstance(metric_data, dict):
            return None
        return metric_data
    
    # 特殊处理：modularity
    if metric_name == 'modularity':
        mod_data = metrics.get('modularity', {})
        if isinstance(mod_data, dict) and 'value' in mod_data:
            return mod_data['value']
        return None
    
    # 其他指标直接返回
    return metrics.get(metric_name)

# Define all metrics to analyze
METRIC_NAMES = [
    'betweenness_centrality',
    'eigenvector_centrality',
    'degree_statistics',  # This will be split into mean_degree, mean_in_degree, mean_out_degree
    'density',
    'clustering',
    'transitivity',
    'modularity',
    'degree_assortativity',
    'connectivity',  # This might need special handling
    'closeness_centrality',
    'global_efficiency',
    'average_shortest_path_length',
    'diameter',
    'k_core',  # This might need special handling
    'algebraic_connectivity',
    'edge_connectivity',
    'node_connectivity'
]

# Additional metrics from degree_statistics
DEGREE_METRICS = ['mean_degree', 'mean_in_degree', 'mean_out_degree']

# Metrics that need special extraction
SPECIAL_METRICS = {
    'k_core': lambda m: m.get('k_core', {}).get('mean_core') if isinstance(m.get('k_core'), dict) else None,
    'connectivity': lambda m: 1 if m.get('connectivity', {}).get('is_weakly_connected', False) else 0,
    'average_shortest_path_length': lambda m: m.get('average_shortest_path_length') if not isinstance(m.get('average_shortest_path_length'), dict) else None,
    'diameter': lambda m: m.get('diameter') if not isinstance(m.get('diameter'), dict) else None,
}


def extract_metric_value_extended(metrics, metric_name):
    """Extended version of extract_metric_value to handle all metrics"""
    # Handle special metrics
    if metric_name in SPECIAL_METRICS:
        return SPECIAL_METRICS[metric_name](metrics)
    
    # Use the original function for standard metrics
    return extract_metric_value(metrics, metric_name)


def load_forget_metrics(jsonl_file):
    """Load max_forgetting scores from JSONL file"""
    forget_data = {}
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                concept = data.get('concept')
                max_forgetting = data.get('avg_forgetting_pct')
                if concept and max_forgetting is not None:
                    forget_data[concept] = max_forgetting
    return forget_data


def load_graph_metrics(json_file):
    """Load graph metrics from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Group by concept (taking the first entry if multiple exist for same concept)
    concept_metrics = {}
    for result in results:
        concept = result.get('concept')
        if concept and concept not in concept_metrics:
            concept_metrics[concept] = result.get('metrics', {})
    
    return concept_metrics


def plot_correlation(forget_scores, graph_metrics, metric_name, output_dir='correlation_plots'):
    """Plot correlation between max_forgetting and a specific graph metric"""
    # Prepare data
    x_values = []  # max_forgetting
    y_values = []  # graph metric
    concepts_used = []
    
    for concept, max_forgetting in forget_scores.items():
        if concept in graph_metrics:
            metrics = graph_metrics[concept]
            
            # Extract metric value
            if metric_name in DEGREE_METRICS:
                # Handle degree_statistics sub-metrics
                value = extract_metric_value(metrics, metric_name)
            elif metric_name in SPECIAL_METRICS:
                # Handle special metrics
                value = extract_metric_value_extended(metrics, metric_name)
            else:
                value = extract_metric_value(metrics, metric_name)
            
            # Skip if value is None, dict (error), or string
            if value is not None and not isinstance(value, (dict, str)):
                try:
                    float_value = float(value)
                    if not np.isnan(float_value) and not np.isinf(float_value):
                        x_values.append(max_forgetting)
                        y_values.append(float_value)
                        concepts_used.append(concept)
                except (ValueError, TypeError):
                    continue
    
    if len(x_values) < 2:
        print(f"Warning: Not enough data points for {metric_name} (only {len(x_values)} points)")
        return
    
    # Check for constant values
    if len(set(y_values)) == 1:
        print(f"Warning: {metric_name} has constant values (all = {y_values[0]}), skipping correlation plot")
        return
    
    # Calculate correlation
    try:
        correlation, p_value = stats.pearsonr(x_values, y_values)
    except Exception as e:
        print(f"Warning: Could not calculate correlation for {metric_name}: {e}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(x_values, y_values, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    # Add trend line
    z = np.polyfit(x_values, y_values, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(x_values), max(x_values), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend line (r={correlation:.3f})')
    
    # Labels and title
    ax.set_xlabel('Max Forgetting Score', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{metric_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    ax.set_title(f'Correlation: Max Forgetting vs {metric_name.replace("_", " ").title()}\n'
                 f'Pearson r = {correlation:.3f}, p-value = {p_value:.3e}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add text box with statistics
    stats_text = f'n = {len(x_values)}\nr = {correlation:.3f}\np = {p_value:.3e}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    safe_metric_name = metric_name.replace('/', '_').replace(' ', '_')
    output_file = os.path.join(output_dir, f'{safe_metric_name}_correlation.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file} (r={correlation:.3f}, n={len(x_values)})")
    plt.close()


def main():
    """Main function"""
    # File paths
    forget_metrics_file = 'data/concept/concept_forget_metrics_v2.jsonl'
    graph_metrics_file = 'results_all_steps_800_5.json'
    output_dir = 'outputs/analysis/correlation_plots/v2'
    
    print("Loading forget metrics...")
    forget_scores = load_forget_metrics(forget_metrics_file)
    print(f"Loaded {len(forget_scores)} concepts with forget scores")
    
    print("Loading graph metrics...")
    graph_metrics = load_graph_metrics(graph_metrics_file)
    print(f"Loaded {len(graph_metrics)} concepts with graph metrics")
    
    # Find common concepts
    common_concepts = set(forget_scores.keys()) & set(graph_metrics.keys())
    print(f"Found {len(common_concepts)} common concepts")
    
    if len(common_concepts) == 0:
        print("Error: No common concepts found between the two files!")
        print(f"Sample forget concepts: {list(forget_scores.keys())[:10]}")
        print(f"Sample graph concepts: {list(graph_metrics.keys())[:10]}")
        return
    
    print(f"Common concepts: {sorted(common_concepts)}")
    
    # Create plots for each metric
    all_metrics = METRIC_NAMES.copy()
    
    # Add degree_statistics sub-metrics
    for degree_metric in DEGREE_METRICS:
        if degree_metric not in all_metrics:
            all_metrics.append(degree_metric)
    
    # Remove 'degree_statistics' from the list since we're using sub-metrics
    if 'degree_statistics' in all_metrics:
        all_metrics.remove('degree_statistics')
    
    print(f"\nGenerating correlation plots for {len(all_metrics)} metrics...")
    print("="*80)
    
    for metric_name in all_metrics:
        try:
            plot_correlation(forget_scores, graph_metrics, metric_name, output_dir)
        except Exception as e:
            print(f"Error plotting {metric_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(f"All plots saved to {output_dir}/")


if __name__ == "__main__":
    main()

