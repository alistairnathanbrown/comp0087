import os
import pandas as pd
import json
import torch
from comet import download_model, load_from_checkpoint
from tqdm import tqdm  # For progress tracking
import time

# Start timing
start_time = time.time()

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
data_file = os.path.join(project_root, 'sts', 'sts_all.csv')
output_dir = os.path.join(project_root, 'metrics', 'comet', 'outputs')
os.makedirs(output_dir, exist_ok=True)

# Load data
print(f"Loading data from {data_file}...")
df = pd.read_csv(data_file, 
                 sep="\\|\\|",  # Use regex escape for pipe characters
                 engine='python',  # Use Python engine for regex separator
                 on_bad_lines='skip')
print(f"Loaded {len(df)} rows of data")

# Load model
print("Downloading COMET model...")
model_path = download_model("Unbabel/wmt22-comet-da")
print("Loading COMET model...")
comet = load_from_checkpoint(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
comet.to(device)

# Check GPU memory
if device.type == "cuda":
    print(f"GPU memory before loading data:")
    print(f" - Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    print(f" - Reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
    print(f" - Max allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")

# Batch processing parameters
batch_size = 128  # Increased for 3090 Ti 24GB GPU
results = []

# Process data in batches
print("Processing data in batches...")
for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
    batch_df = df.iloc[i:i+batch_size]
    batch_data = []
    
    # Prepare batch inputs
    for _, row in batch_df.iterrows():
        s1 = row['s1']
        s2 = row['s2']
        batch_data.append({
            "src": "",  # Source sentence is not used in this case
            "mt": s1,
            "ref": s2, 
        })
    
    # Run prediction on batch
    batch_scores = comet.predict(batch_data)
    
    # Monitor GPU memory usage after batch processing (every 5 batches)
    if device.type == "cuda" and i % (batch_size * 5) == 0:
        print(f"GPU memory after batch {i//batch_size}:")
        print(f" - Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f" - Max allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
    
    # Store results
    for j, (_, row) in enumerate(batch_df.iterrows()):
        results.append({
            's1': row['s1'], 
            's2': row['s2'], 
            'sts_score': row['score'], 
            'metric_score': batch_scores["scores"][j]
        })
    
    # Save intermediate results every 10 batches
    if i % (batch_size * 10) == 0 and i > 0:
        print(f"Processed {i}/{len(df)} rows. Saving intermediate results...")
        temp_df = pd.DataFrame(results)
        temp_df.to_csv(os.path.join(output_dir, f'comet_scores_sts_temp_{i}.csv'), index=False, sep="|")

# Create final DataFrame and save results
results_df = pd.DataFrame(results)
csv_output_path = os.path.join(output_dir, 'comet_scores_sts.csv')
results_df.to_csv(csv_output_path, index=False, sep="|")

json_output_path = os.path.join(output_dir, 'comet_scores_sts.json')
with open(json_output_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

elapsed_time = time.time() - start_time
print(f"Processing completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print("Scores have been saved to:")
print(f" - CSV: {csv_output_path}")
print(f" - JSON: {json_output_path}")