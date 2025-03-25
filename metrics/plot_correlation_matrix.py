import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

bleu_df = pd.read_csv('bleu/outputs/bleu_scores_sts.csv', sep='|') 
bleurt_df = pd.read_csv('bleurt/outputs/bleurt_scores_sts.csv', sep='|') 
chrf_df = pd.read_csv('chrf++/outputs/chrF_scores_sts.csv', sep='|') 
comet_df = pd.read_csv('comet/outputs/comet_scores_sts.csv', sep='|')
# normalise bleu
bleu_df['metric_score'] = bleu_df['metric_score'] / 100
bleu_df = bleu_df.rename(columns={'metric_score': 'bleu_score'})

# normalise bleurt (clip all past 0 to and past 1 to 1)
bleurt_df['bleurt_score'] = bleurt_df['bleurt_score'].clip(0, 1)

# normalise chrf++
chrf_df['metric_score'] = chrf_df['metric_score'] / 100
chrf_df = chrf_df.rename(columns={'metric_score': 'chrf_score'})

comet_df = comet_df.rename(columns={'metric_score': 'comet_score'})

merged_df = bleu_df.merge(bleurt_df, on=['s1', 's2', 'sts_score'], how='outer').merge(chrf_df, on=['s1', 's2', 'sts_score'], how='outer').merge(comet_df, on=['s1', 's2', 'sts_score'], how='outer')

# Select only numeric columns
numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
numeric_df = merged_df[numeric_columns]

# Calculate correlation matrix
correlation_matrix = numeric_df.corr().round(2)

# Plot heatmap
plt.figure(figsize=(10, 8)) 
plt.tight_layout()
sns.heatmap(correlation_matrix, annot=True, cmap="rocket", square=True)
plt.savefig('matrix.png')
plt.savefig('matrix.pdf')


# List of metrics to compare with STS
metrics = ['bleu_score', 'bleurt_score', 'chrf_score', 'comet_score']

# Calculate different correlation coefficients
results = []
for metric in metrics:
    pearson = numeric_df['sts_score'].corr(numeric_df[metric], method='pearson')
    kendall = numeric_df['sts_score'].corr(numeric_df[metric], method='kendall')
    spearman = numeric_df['sts_score'].corr(numeric_df[metric], method='spearman')
    results.append([metric, pearson, kendall, spearman])

# Create a DataFrame from the results
correlation_table = pd.DataFrame(results, columns=['Metric', 'Pearson', 'Kendall', 'Spearman'])

# Round the values to 3 decimal places
correlation_table = correlation_table.round(3)

# Print the table
print(correlation_table.to_string(index=False))

# Optionally, save the table to a CSV file
correlation_table.to_csv('sts_correlation_table.csv', index=False)

correlation_table.to_latex('sts_correlation_table.tex', index=False, float_format="%.3f", caption="Correlation coefficients between STS and various metrics.", label="tab:correlation_table")