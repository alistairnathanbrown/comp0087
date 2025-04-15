import os
import json
import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL_OUTPUT = os.path.join(script_dir, "../../results/idiom_translation_Meta-Llama-3.1-8B-Instruct_20250415_122843.json")
INCONTEXT_MODEL_OUTPUT = os.path.join(script_dir, "../../results/idiom_translation_Meta-Llama-3.1-8B-Instruct_20250415_115356.json")
RAG_MODEL_OUTPUT = os.path.join(script_dir, "../../utils/idiom_translation_from_examples.csv")
COTA_MODEL_OUTPUT = os.path.join(script_dir, "../../results/cot_a_with_id.json")
COTB_MODEL_OUTPUT = os.path.join(script_dir, "../../results/cot_b_with_id.json")

DATASET = os.path.join(script_dir, "../../idiom_data/idiom_dataset_en.csv")


def load_json_translations(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {entry['id']: entry['translated_sentence'] for entry in data.get("translations", [])}

base_model_data = load_json_translations(BASE_MODEL_OUTPUT)
incontext_model_data = load_json_translations(INCONTEXT_MODEL_OUTPUT)
cotA_model_data = load_json_translations(COTA_MODEL_OUTPUT)
cotB_model_data = load_json_translations(COTB_MODEL_OUTPUT)

rag_df = pd.read_csv(RAG_MODEL_OUTPUT, encoding='utf-8')
rag_model_data = pd.Series(rag_df.translated_sentence.values, index=rag_df.id).to_dict()

correct_df = pd.read_csv(DATASET, encoding='utf-8')
correct_translations = pd.Series(correct_df.good_paraphrase.values, index=correct_df.id).to_dict()
idiom_sentences = pd.Series(correct_df.idiom_sentence.values, index=correct_df.id).to_dict()


all_ids = set(list(base_model_data.keys()) +
              list(incontext_model_data.keys()) +
              list(rag_model_data.keys()) +
              list(cotA_model_data.keys()) +
              list(cotB_model_data.keys()) +
              list(correct_translations.keys()))

rows = []
for id_val in sorted(all_ids):
    row = {
        'id': id_val,
        'base_model_output': base_model_data.get(id_val, ""),
        'incontext_model_output': incontext_model_data.get(id_val, ""),
        'rag_model_output': rag_model_data.get(id_val, ""),
        'cotA_model_output': cotA_model_data.get(id_val, ""),
        'cotB_model_output': cotB_model_data.get(id_val, ""),
        'correct_translation': correct_translations.get(id_val, ""),
        'idiom_sentence': idiom_sentences.get(id_val, "")
    }
    rows.append(row)

results_df = pd.DataFrame(rows)

model_path = download_model("Unbabel/wmt22-comet-da")
comet = load_from_checkpoint(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
comet.to(device)

def compute_comet_score(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return None
    data = [{
        "src": "",  # Source sentence is not used in this case
        "mt": hypothesis,
        "ref": reference
    }]
    scores = comet.predict(data)
    return scores[0]

#comet scores against good_paraphrase
comet_base = []
comet_incontext = []
comet_rag = []
comet_cotA = []
comet_cotB = []

#comet scores against original idiomatic
comet_base_idiom = []
comet_incontext_idiom = []
comet_rag_idiom = []
comet_cotA_idiom = []
comet_cotB_idiom = []
comet_correct_idiom = []

for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Processing rows"):
    # Reference for first set: correct translation (good_paraphrase)
    ref_good = row['correct_translation']

    # Compute COMET scores where reference is the good translation
    base_score = compute_comet_score(row['base_model_output'], ref_good) if row['base_model_output'] and ref_good else None
    incontext_score = compute_comet_score(row['incontext_model_output'], ref_good) if row['incontext_model_output'] and ref_good else None
    rag_score = compute_comet_score(row['rag_model_output'], ref_good) if row['rag_model_output'] and ref_good else None
    cotA_score = compute_comet_score(row['cotA_model_output'], ref_good) if row['cotA_model_output'] and ref_good else None
    cotB_score = compute_comet_score(row['cotB_model_output'], ref_good) if row['cotB_model_output'] and ref_good else None

    comet_base.append(base_score)
    comet_incontext.append(incontext_score)
    comet_rag.append(rag_score)
    comet_cotA.append(cotA_score)
    comet_cotB.append(cotB_score)

    # Reference for second set: idiom_sentence from the CSV
    ref_idiom = row['idiom_sentence']

    # Compute COMET scores using idiom_sentence as reference.
    base_score_idiom = compute_comet_score(row['base_model_output'], ref_idiom) if row['base_model_output'] and ref_idiom else None
    incontext_score_idiom = compute_comet_score(row['incontext_model_output'], ref_idiom) if row['incontext_model_output'] and ref_idiom else None
    rag_score_idiom = compute_comet_score(row['rag_model_output'], ref_idiom) if row['rag_model_output'] and ref_idiom else None
    cotA_score_idiom = compute_comet_score(row['cotA_model_output'], ref_idiom) if row['cotA_model_output'] and ref_idiom else None
    cotB_score_idiom = compute_comet_score(row['cotB_model_output'], ref_idiom) if row['cotB_model_output'] and ref_idiom else None
    correct_score_idiom = compute_comet_score(row['correct_translation'], ref_idiom) if row['correct_translation'] and ref_idiom else None

    comet_base_idiom.append(base_score_idiom)
    comet_incontext_idiom.append(incontext_score_idiom)
    comet_rag_idiom.append(rag_score_idiom)
    comet_cotA_idiom.append(cotA_score_idiom)
    comet_cotB_idiom.append(cotB_score_idiom)
    comet_correct_idiom.append(correct_score_idiom)


results_df['comet_base'] = comet_base
results_df['comet_incontext'] = comet_incontext
results_df['comet_rag'] = comet_rag
results_df['comet_cotA'] = comet_cotA
results_df['comet_cotB'] = comet_cotB

results_df['comet_base_idiom'] = comet_base_idiom
results_df['comet_incontext_idiom'] = comet_incontext_idiom
results_df['comet_rag_idiom'] = comet_rag_idiom
results_df['comet_cotA_idiom'] = comet_cotA_idiom
results_df['comet_cotB_idiom'] = comet_cotB_idiom
results_df['comet_correct_idiom'] = comet_correct_idiom

output_csv = 'combined_translations_and_comet_scores.csv'
results_df.to_csv(output_csv, index=False, encoding='utf-8')
print(f"Results saved to {output_csv}")