import os
import pandas as pd
import json
import torch
from comet import download_model, load_from_checkpoint

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
data_file = os.path.join(project_root, 'idiom_data', 'idiom_dataset.csv')

output_dir = os.path.join(project_root, 'metrics', 'comet', 'outputs')
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(data_file)

model_path = download_model("Unbabel/wmt22-comet-da")
comet = load_from_checkpoint(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
comet.to(device)
results = []

for _, row in df.iterrows():
    id_val = row['id']
    hypothesis = row['idiom_sentence']
    reference = row['good_paraphrase']

    data = [{
        "src": "",  # Source sentence is not used in this case
        "mt": hypothesis,
        "ref": reference
    }]
    
    scores = comet.predict(data)
    score = scores["scores"][0]
    results.append({'id': id_val, 'score': score})

results_df = pd.DataFrame(results)

csv_output_path = os.path.join(output_dir, 'comet_scores.csv')
results_df.to_csv(csv_output_path, index=False)

json_output_path = os.path.join(output_dir, 'comet_scores.json')
with open(json_output_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Scores have been saved to:")
print(f" - CSV: {csv_output_path}")
print(f" - JSON: {json_output_path}")
