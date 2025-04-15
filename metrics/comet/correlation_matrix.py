import os
import pandas as pd
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

# Set up file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
input_file = os.path.join(project_root, 'metrics', 'comet', 'outputs', 'comet_scores_sts.csv')
output_dir = os.path.join(project_root, 'metrics', 'comet', 'outputs')

# Read the CSV file with single pipe delimiter
print(f"Reading data from {input_file}...")
df = pd.read_csv(input_file, 
                 sep='|', 
                 engine='python',
                 on_bad_lines='skip')

# Print column information for debugging
print(f"Columns found: {df.columns.tolist()}")
print(f"First few rows:\n{df.head()}")

# Convert numeric columns if needed
for col in ['sts_score', 'metric_score']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Add normalized metric score and delta columns
print("Calculating normalized scores and deltas...")
df['normalized_metric_score'] = df['metric_score'] / 100
df['delta'] = abs(df['sts_score'] - df['normalized_metric_score'])

# Calculate statistics
avg_delta = df['delta'].mean()
min_delta = df['delta'].min()
max_delta = df['delta'].max()
median_delta = df['delta'].median()

# Find pairs with min and max deltas
min_delta_row = df.loc[df['delta'].idxmin()]
max_delta_row = df.loc[df['delta'].idxmax()]

# Calculate correlation matrix
print("\nCalculating correlation matrix...")
correlation_matrix = df[['sts_score', 'metric_score', 'normalized_metric_score']].corr()
print("Correlation matrix:")
print(correlation_matrix)

# Calculate different correlation metrics between STS and normalized metric scores
pearson_corr, pearson_p = pearsonr(df['sts_score'], df['normalized_metric_score'])
spearman_corr, spearman_p = spearmanr(df['sts_score'], df['normalized_metric_score'])
kendall_corr, kendall_p = kendalltau(df['sts_score'], df['normalized_metric_score'])

print("\nCorrelation between STS scores and normalized comet scores:")
print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.10e})")
print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.10e})")
print(f"Kendall's tau: {kendall_corr:.4f} (p-value: {kendall_p:.10e})")

# Print evaluation results
print("\n===== Evaluation Results =====")
print(f"Average delta: {avg_delta:.4f}")
print(f"Min delta: {min_delta:.4f}")
print(f"Max delta: {max_delta:.4f}")
print(f"Median delta: {median_delta:.4f}")

print("\nBest matching pair (smallest delta):")
print(f"S1: \"{min_delta_row['s1']}\"")
print(f"S2: \"{min_delta_row['s2']}\"")
print(f"STS score: {min_delta_row['sts_score']:.2f}")
print(f"Normalized metric score: {min_delta_row['normalized_metric_score']:.2f}")
print(f"Delta: {min_delta_row['delta']:.4f}")

print("\nWorst matching pair (largest delta):")
print(f"S1: \"{max_delta_row['s1']}\"")
print(f"S2: \"{max_delta_row['s2']}\"")
print(f"STS score: {max_delta_row['sts_score']:.2f}")
print(f"Normalized metric score: {max_delta_row['normalized_metric_score']:.2f}")
print(f"Delta: {max_delta_row['delta']:.4f}")

# Save results to CSV with the same single pipe delimiter
enhanced_output_path = os.path.join(output_dir, 'comet_scores_sts_with_delta.csv')
df.to_csv(enhanced_output_path, index=False, sep='|')

# Save evaluation summary including correlations
summary = {
    "average_delta": float(avg_delta),
    "min_delta": float(min_delta),
    "max_delta": float(max_delta),
    "median_delta": float(median_delta),
    "correlations": {
        "pearson": {
            "coefficient": float(pearson_corr),
            "p_value": float(pearson_p)
        },
        "spearman": {
            "coefficient": float(spearman_corr),
            "p_value": float(spearman_p)
        },
        "kendall": {
            "coefficient": float(kendall_corr),
            "p_value": float(kendall_p)
        }
    },
    "best_match": {
        "s1": min_delta_row['s1'],
        "s2": min_delta_row['s2'],
        "sts_score": float(min_delta_row['sts_score']),
        "normalized_metric_score": float(min_delta_row['normalized_metric_score']),
        "delta": float(min_delta_row['delta'])
    },
    "worst_match": {
        "s1": max_delta_row['s1'],
        "s2": max_delta_row['s2'],
        "sts_score": float(max_delta_row['sts_score']),
        "normalized_metric_score": float(max_delta_row['normalized_metric_score']),
        "delta": float(max_delta_row['delta'])
    }
}

summary_path = os.path.join(output_dir, 'comet_evaluation_summary.json')
with open(summary_path, 'w') as summary_file:
    json.dump(summary, summary_file, indent=4)

print("\nFiles saved:")
print(f" - Enhanced CSV: {enhanced_output_path}")
print(f" - Summary: {summary_path}")

# Optional: Create a scatter plot of STS vs normalized metric scores
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(df['sts_score'], df['normalized_metric_score'], alpha=0.6)
    
    # Add trend line
    z = np.polyfit(df['sts_score'], df['normalized_metric_score'], 1)
    p = np.poly1d(z)
    plt.plot(df['sts_score'], p(df['sts_score']), "r--", alpha=0.8)
    
    # Add perfect correlation line for reference
    plt.plot([0, 1], [0, 1], 'g-', alpha=0.3)
    
    # Add correlation coefficients as text
    plt.text(0.05, 0.95, f"Pearson: {pearson_corr:.4f}", transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, f"Spearman: {spearman_corr:.4f}", transform=plt.gca().transAxes)
    plt.text(0.05, 0.85, f"Kendall: {kendall_corr:.4f}", transform=plt.gca().transAxes)
    
    # Formatting
    plt.xlabel('STS Score (Human Judgment)')
    plt.ylabel('Normalized comet Score')
    plt.title('Correlation between Human Judgments and comet Metric')
    plt.grid(True, alpha=0.3)
    
    # Set both axes to same scale (0 to 1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add grid lines at 0.1 intervals
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    
    # Save the plot
    correlation_plot_path = os.path.join(output_dir, 'sts_vs_comet_correlation.png')
    plt.savefig(correlation_plot_path)
    print(f" - Correlation plot: {correlation_plot_path}")
    
    # Create histogram of deltas
    plt.figure(figsize=(10, 6))
    plt.hist(df['delta'], bins=20, edgecolor='black')
    plt.title('Distribution of Deltas between STS and Normalized comet Scores')
    plt.xlabel('Delta (absolute difference)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    hist_path = os.path.join(output_dir, 'delta_histogram.png')
    plt.savefig(hist_path)
    print(f" - Histogram: {hist_path}")
    
except ImportError:
    print("Matplotlib not available. Skipping visualization.")
except Exception as e:
    print(f"Error creating visualizations: {e}")