import os
import pandas as pd
import json
from sacrebleu.metrics import CHRF

os.makedirs('outputs', exist_ok=True)

df = pd.read_csv('idiom_data/idiom_dataset.csv')

chrf = CHRF()
results = []

for _, row in df.iterrows():
    id_val = row['id']
    hypothesis = row['idiom_sentance']
    reference = row['good_paraphrase']
    score = chrf.sentence_score(hypothesis, [reference]).score
    results.append({'id': id_val, 'score': score})

results_df = pd.DataFrame(results)

csv_output_path = os.path.join('outputs', 'chrF_scores.csv')
results_df.to_csv(csv_output_path, index=False)

json_output_path = os.path.join('outputs', 'chrF_scores.json')
with open(json_output_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Scores have been saved to:")
print(f" - CSV: {csv_output_path}")
print(f" - JSON: {json_output_path}")