import json
import os

def copy_ids_in_order(base_json_path, new_json_path, output_json_path):
    # 1) Load the base JSON
    with open(base_json_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    # 2) Load the new JSON
    with open(new_json_path, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    base_translations = base_data["translations"]
    new_translations = new_data["translations"]

    # 3) Ensure they have the same number of sentences
    if len(base_translations) != len(new_translations):
        raise ValueError("Base and new JSON files do not have the same number of translations.")

    # 4) Loop and copy IDs
    for i in range(len(base_translations)):
        new_translations[i]["id"] = base_translations[i]["id"]

    # 5) Save the updated JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

# Example usage:
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_json_path = os.path.join(script_dir, "results/idiom_translation_Meta-Llama-3.1-8B-Instruct_20250415_122843.json")
    new_json_path = os.path.join(script_dir, "results/idiom_translation_Meta-Llama-3.1-8B-Instruct_20250414_231142_variant_prompt_b.json")
    output_json_path = os.path.join(script_dir, "results/cot_b_with_id.json")

    copy_ids_in_order(base_json_path, new_json_path, output_json_path)
    print(f"Updated file saved to {output_json_path}")
