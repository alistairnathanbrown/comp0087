import os
import pandas as pd
import json
from sacrebleu.metrics import BLEU

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
data_file = os.path.join(project_root, 'idiom_data', 'idiom_dataset.csv')

output_dir = os.path.join(project_root, 'metrics', 'bleu', 'outputs')
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(data_file)

bleu = BLEU(effective_order=True)
results = []

for _, row in df.iterrows():
    id_val = row['id']
    hypothesis = row['idiom_sentence']
    reference = row['good_paraphrase']
    score = bleu.sentence_score(hypothesis, [reference]).score
    results.append({'id': id_val, 'score': score})

results_df = pd.DataFrame(results)

csv_output_path = os.path.join(output_dir, 'bleu_scores.csv')
results_df.to_csv(csv_output_path, index=False)

json_output_path = os.path.join(output_dir, 'bleu_scores.json')
with open(json_output_path, 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Scores have been saved to:")
print(f" - CSV: {csv_output_path}")
print(f" - JSON: {json_output_path}")