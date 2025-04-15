import os
import pandas as pd
import json
import numpy as np

# Set up file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
input_file = os.path.join(project_root, 'metrics', 'bleu', 'outputs', 'bleu_scores_sts.csv')
output_dir = os.path.join(project_root, 'metrics', 'bleu', 'outputs')

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
enhanced_output_path = os.path.join(output_dir, 'chrF_scores_sts_with_delta.csv')
df.to_csv(enhanced_output_path, index=False, sep='|')

# Save evaluation summary
summary = {
    "average_delta": float(avg_delta),
    "min_delta": float(min_delta),
    "max_delta": float(max_delta),
    "median_delta": float(median_delta),
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

summary_path = os.path.join(output_dir, 'chrF_evaluation_summary.json')
with open(summary_path, 'w') as summary_file:
    json.dump(summary, summary_file, indent=4)

print("\nFiles saved:")
print(f" - Enhanced CSV: {enhanced_output_path}")
print(f" - Summary: {summary_path}")

# Optional: Create a histogram of deltas
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(df['delta'], bins=20, edgecolor='black')
    plt.title('Distribution of Deltas between STS and Normalized bleu Scores')
    plt.xlabel('Delta (absolute difference)')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    
    hist_path = os.path.join(output_dir, 'delta_histogram.png')
    plt.savefig(hist_path)
    print(f" - Histogram: {hist_path}")
except ImportError:
    print("Matplotlib not available. Skipping histogram generation.")