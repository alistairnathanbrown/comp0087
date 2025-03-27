import os
import pandas as pd
import json
from sacrebleu.metrics import CHRF

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
data_file = os.path.join(project_root, 'sts', 'sts_all.csv')

output_dir = os.path.join(project_root, 'metrics', 'chrf++', 'outputs')
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(data_file, 
                 sep="\\|\\|",  # Use regex escape for pipe characters
                 engine='python',  # Use Python engine for regex separator
                 on_bad_lines='skip')

chrf = CHRF()
results = []

for _, row in df.iterrows():
    # id_val = row['id']
    s1= row['s1']
    s2= row['s2']
    sts_score = row['score']
    score = chrf.sentence_score(s1, [s2]).score
    results.append({'s1': s1, 's2': s2, 'sts_score': sts_score, 'metric_score': score})

results_df = pd.DataFrame(results)

csv_output_path = os.path.join(output_dir, 'chrF_scores_sts.csv')
results_df.to_csv(csv_output_path, index=False, sep="|")

json_output_path = os.path.join(output_dir, 'chrF_scores_sts.json')
with open(json_output_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Scores have been saved to:")
print(f" - CSV: {csv_output_path}")
print(f" - JSON: {json_output_path}")