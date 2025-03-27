import os
import pandas as pd
import json
from evaluate import load
from tqdm import tqdm
import time

# Start timing
start_time = time.time()

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
data_file = os.path.join(project_root, 'sts', 'sts_all.csv')
output_dir = os.path.join(project_root, 'metrics', 'bleurt', 'outputs')
os.makedirs(output_dir, exist_ok=True)

# Load data
print(f"Loading data from {data_file}...")
df = pd.read_csv(data_file,
                 sep="\\|\\|",
                 engine='python',
                 on_bad_lines='skip')
print(f"Loaded {len(df)} rows of data")

# Load BLEURT metric
print("Loading BLEURT metric...")
bleurt = load("bleurt", module_type="metric")
print("BLEURT metric loaded successfully")

# Batch processing parameters
batch_size = 128
results = []

# Process data in batches
print("Processing data in batches...")
for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
    batch_df = df.iloc[i:i+batch_size]
    
    # Prepare batch inputs
    candidates = batch_df['s1'].tolist()
    references = batch_df['s2'].tolist()
    
    # Compute BLEURT scores
    batch_scores = bleurt.compute(predictions=candidates, references=references)['scores']
    
    # Store results
    for j, (_, row) in enumerate(batch_df.iterrows()):
        results.append({
            's1': row['s1'],
            's2': row['s2'],
            'sts_score': row['score'],
            'bleurt_score': batch_scores[j]
        })
    
    # Save intermediate results every 10 batches
    if i % (batch_size * 10) == 0 and i > 0:
        print(f"\nSaving intermediate results after {i} rows...")
        temp_df = pd.DataFrame(results)
        temp_path = os.path.join(output_dir, f'bleurt_scores_sts_temp_{i}.csv')
        temp_df.to_csv(temp_path, index=False, sep="|")

# Save final results
print("\nSaving final results...")
results_df = pd.DataFrame(results)
csv_output_path = os.path.join(output_dir, 'bleurt_scores_sts.csv')
results_df.to_csv(csv_output_path, index=False, sep="|")

json_output_path = os.path.join(output_dir, 'bleurt_scores_sts.json')
with open(json_output_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

# Print completion message
elapsed_time = time.time() - start_time
print(f"\n{'='*40}\nProcessing completed in {elapsed_time/60:.2f} minutes")
print("Final outputs saved to:")
print(f" - CSV: {csv_output_path}")
print(f" - JSON: {json_output_path}\n{'='*40}")
