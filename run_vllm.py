import argparse
import csv
import json
import os
import pandas as pd
from datetime import datetime
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

def translate_idioms(model_name, csv_path, temperature=0.7, max_tokens=1024, checkpoint_interval=50, checkpoint_dir="checkpoints"):
    """
    Translates idiomatic expressions in sentences using the specified model.
    
    Args:
        model_name: The name or path of the LLM model to use
        csv_path: Path to the CSV file containing idiom data
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        checkpoint_interval: Save results after processing this many sentences
        checkpoint_dir: Directory to save checkpoint files
        
    Returns:
        A list of dictionaries containing the original and translated sentences
    """
    print(f"Initializing model: {model_name}")
    llm = LLM(
        model=model_name,
        max_model_len=40816,  # Consider making this a parameter too
        gpu_memory_utilization=0.95,
    )
    
    # Define the expected JSON structure for guided decoding
    json_schema = {
        "type": "object",
        "properties": {
            "original_sentence": {"type": "string"},
            "translated_sentence": {"type": "string"}
        },
        "required": ["original_sentence", "translated_sentence"]
    }
    
    # Load the dataset
    print(f"Reading data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        if "original_idiom_sentence" not in df.columns:
            raise ValueError(f"CSV file must contain 'original_idiom_sentence' column. Found columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []
    
    # System prompt for idiom translation
    #content = """You are a translator specializing in idiomatic expressions. 

    #You will be given a sentence with an idiom marked between ID tags like IDidiomID.
    #Your task is to replace the idiom with a NON-FIGURATIVE, LITERAL explanation of what the idiom means.

    #IMPORTANT: Do NOT simply add spaces between words or make minor changes - you must completely replace the idiom with its literal, non-idiomatic meaning.

    #For example:
    #- If you see "IDraining catsID and dogs" you might translate it to "raining very heavily"
    #- If you see "IDkick the bucketID" you might translate it to "die"
    #- If you see "IDbananarepublicID" you might translate it to "politically unstable country with corrupt government"

    #Your response must be a valid JSON object with two fields:
    #1. 'original_sentence' (the exact input)
    #2. 'translated_sentence' (your translation with the idiom replaced with its literal meaning)

    #Keep the overall sentence structure intact, changing only the idiom."""

    #cot_content = """You are a translator specializing in idiomatic expressions. For the input sentence, the idiom is marked between ID tags (e.g., IDidiomID).

    #You will be given a sentence with an idiom marked between ID tags like IDidiomID.
    #Your task is to replace the idiom with a NON-FIGURATIVE, LITERAL explanation of what the idiom means.

    #IMPORTANT: Do NOT simply add spaces between words or make minor changes - you must completely replace the idiom with its literal, non-idiomatic meaning.

    #For example:
    #- If you see "IDraining catsID and dogs" you might translate it to "raining very heavily"
    #- If you see "IDkick the bucketID" you might translate it to "die"
    #- If you see "IDbananarepublicID" you might translate it to "politically unstable country with corrupt government"

    #Please follow these steps internally:
    #1. Explain both the literal and figurative meanings of the idiom, including any relevant cultural or contextual nuances.
    #2. Provide a direct, word-for-word literal translation of the idiom.
    #3. Modify this literal translation to capture the idiom’s intended meaning naturally in <LANGUAGE>, taking into account common expressions and cultural context.
    #4. Review your refined translation to check for any discrepancies with standard usage.

    #Finally, replace the idiom in the original sentence with your final translation. Your answer should be a valid JSON object with two fields:
    #- "original_sentence": the exact input sentence.
    #- "translated_sentence": the sentence with the idiom replaced by its literal explanation.

    #Do not include any of your internal reasoning in your final output."""

    #cot_content = """You are an expert translator with a deep understanding of idiomatic expressions and their cultural contexts. For the input sentence, the idiom is marked between ID tags (e.gIDidiomID).

    #Your task is to replace the idiom with a NON-FIGURATIVE, LITERAL explanation of what the idiom means.

    #IMPORTANT: Do NOT simply add spaces between words or make minor changes - you must completely replace the idiom with its literal, non-idiomatic meaning. 

    #For example:
    #- If you see "IDraining catsID and dogs" you might translate it to "raining very heavily"
    #- If you see "IDkick the bucketID" you might translate it to "die"
    #- If you see "IDbananarepublicID" you might translate it to "politically unstable country with corrupt government"

    #Internally, perform the following steps:
    #1. Analyze the context of the idiom by identifying any cultural references, common usage patterns, and connotations beyond the literal words.
    #2. Break down the idiom into its basic components, providing a straightforward, literal interpretation of each element.
    #3. Synthesize these insights to construct a final, non-figurative, literal explanation of what the idiom means.
    #4. Cross-check the resulting explanation to ensure that it accurately conveys the intended meaning without retaining any figurative language.

    #Finally, replace the idiom in the original sentence with your final translation. Your answer should be a valid JSON object with two fields:
    #- "original_sentence": the exact input sentence.
    #- "translated_sentence": the sentence with the idiom replaced by its literal explanation.

    #Do not include any of your internal reasoning in your final output."""

    cot_content = """You are a translator with expert knowledge of idiomatic expressions and cultural nuances. For the input sentence, the idiom is enclosed in ID tags (e.g., IDidiomID). Your task is to replace the idiom with a non-figurative, literal explanation of its meaning.

    Your task is to replace the idiom with a NON-FIGURATIVE, LITERAL explanation of what the idiom means. 

    IMPORTANT: Do NOT simply add spaces between words or make minor changes - you must completely replace the idiom with its literal, non-idiomatic meaning.

    For example:
    - If you see "IDraining catsID and dogs" you might translate it to "raining very heavily"
    - If you see "IDkick the bucketID" you might translate it to "die"
    - If you see "IDbananarepublicID" you might translate it to "politically unstable country with corrupt government" 

    Internally, please perform the following steps:
    1. Analyze the idiom by describing both its literal and figurative meanings, noting any cultural or contextual details.
    2. Generate an initial, straightforward literal translation of the idiom.
    3. Review the initial translation for clarity and correctness; refine it to ensure that it fully captures the intended meaning without any figurative language.
    4. Confirm that your final refined translation is complete and accurate.

    Finally, replace the idiom in the original sentence with your final refined translation. Your answer should be a valid JSON object with two fields:
    - "original_sentence": the exact input sentence.
    - "translated_sentence": the sentence where the idiom has been replaced with its literal explanation.

    Do not include any internal reasoning or draft steps in your final output."""

    guided_decoding_params = GuidedDecodingParams(json=json_schema, backend="auto")
    
    # Add sampling parameters with guided decoding
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|eot_id|>"],
        guided_decoding=guided_decoding_params
    )
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Generate a unique run ID based on model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = model_name.split('/')[-1]
    run_id = f"{model_short_name}_{timestamp}"
    
    # Check for existing checkpoint to resume from
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{run_id}.json")
    checkpoint_info_file = os.path.join(checkpoint_dir, f"checkpoint_info_{run_id}.json")
    
    # Initialize results and starting index
    results = []
    start_idx = 0
    
    # Save checkpoint information
    with open(checkpoint_info_file, 'w', encoding='utf-8') as f:
        json.dump({
            "run_id": run_id,
            "model": model_name,
            "total_sentences": len(df),
            "start_time": timestamp,
            "status": "started"
        }, f, indent=2)
    
    total = len(df)
    print(f"Processing {total} sentences...")
    
    # Process each sentence
    for idx, (df_idx, row) in enumerate(df.iterrows()):
        # Skip already processed items if resuming
        if idx < start_idx:
            continue
            
        try:
            original_sentence = row["original_idiom_sentence"]
            print(f"Processing {idx+1}/{total}: {original_sentence[:50]}...")
            
            # Create conversation for the model
            conversation = [
                {
                    "role": "system",
                    "content": cot_content,
                },
                {
                    "role": "user",
                    "content": original_sentence    
                },
            ]
            
            # Generate translation
            outputs = llm.chat(conversation, sampling_params=sampling_params)
               
            for output in outputs:
                full_text = output.outputs[0].text
                try:
                    json_output = json.loads(full_text)
                    results.append(json_output)
                    print(f"✓ Successfully processed")
                except json.JSONDecodeError as e:
                    print(f"× Error parsing JSON: {e}")
                    # Store the error case with the original text
                    results.append({
                        "original_sentence": original_sentence,
                        "translated_sentence": "ERROR: Could not parse model output",
                        "raw_output": full_text,
                        "error": str(e)
                    })
                
                # Save checkpoint at specified intervals
                if (idx + 1) % checkpoint_interval == 0:
                    print(f"\nCheckpoint: Saving progress after {idx + 1}/{total} sentences...")
                    save_checkpoint(results, run_id, checkpoint_dir, idx + 1, total)
                    print(f"Checkpoint saved. Continuing processing...")
                
        except Exception as e:
            print(f"× Error processing sentence: {e}")
            results.append({
                "original_sentence": original_sentence if 'original_sentence' in locals() else "Unknown",
                "translated_sentence": f"ERROR: {str(e)}",
                "error": str(e)
            })
    
    # Save final checkpoint
    if results:
        print(f"\nSaving final checkpoint after completing all {total} sentences...")
        save_checkpoint(results, run_id, checkpoint_dir, total, total, is_final=True)
    
    return results, run_id

def save_checkpoint(results, run_id, checkpoint_dir, current_idx, total, is_final=False):
    """
    Saves checkpoint of current progress
    
    Args:
        results: List of dictionaries with translation results so far
        run_id: Unique identifier for this run
        checkpoint_dir: Directory to save checkpoints
        current_idx: Current index in processing
        total: Total number of items to process
        is_final: Whether this is the final checkpoint
    """
    # Create checkpoint filename
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{run_id}.json")
    checkpoint_info_file = os.path.join(checkpoint_dir, f"checkpoint_info_{run_id}.json")
    
    # Save results checkpoint
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump({
            "run_id": run_id,
            "current_index": current_idx,
            "total": total,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "translations": results
        }, f, indent=2, ensure_ascii=False)
    
    # Update checkpoint info
    with open(checkpoint_info_file, 'w', encoding='utf-8') as f:
        json.dump({
            "run_id": run_id,
            "current_index": current_idx,
            "total_sentences": total,
            "progress_percentage": round((current_idx / total) * 100, 2),
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "completed" if is_final else "in_progress"
        }, f, indent=2)
    
    # If this is a final checkpoint, also create a "latest completed" pointer
    if is_final:
        latest_file = os.path.join(checkpoint_dir, "latest_completed.txt")
        with open(latest_file, 'w') as f:
            f.write(run_id)
    
    return checkpoint_file

def save_results(results, model_name, run_id=None):
    """
    Saves results to both JSON and CSV formats
    
    Args:
        results: List of dictionaries with original and translated sentences
        model_name: Name of the model used (for filename)
        run_id: Optional unique run identifier (if None, a new one is created)
    """
    # Create a timestamp and run_id if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = model_name.split('/')[-1]
    
    if run_id is None:
        run_id = f"{model_short_name}_{timestamp}"
    
    # Create output directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save as JSON
    json_filename = f"results/idiom_translation_{run_id}_variant_prompt_c.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump({
            "model": model_name,
            "timestamp": timestamp,
            "translations": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Saved JSON results to {json_filename}")
    
    # Save as CSV
    csv_filename = f"results/idiom_translation_{run_id}_variant_prompt_c.csv"
    
    # Flatten any nested structures for CSV
    flat_results = []
    for item in results:
        flat_item = {
            "original_sentence": item.get("original_sentence", ""),
            "translated_sentence": item.get("translated_sentence", "")
        }
        # Add any other fields that might be in the results
        for k, v in item.items():
            if k not in flat_item and k != "translated_sentence" and k != "original_sentence":
                flat_item[k] = v
        flat_results.append(flat_item)
    
    # Write CSV
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        #if flat_results:
        #    writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())
        #    writer.writeheader()
        #    writer.writerows(flat_results)
        fieldnames = ["original_sentence", "translated_sentence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
          filtered_item = {
            "original_sentence": item.get("original_sentence", ""),
            "translated_sentence": item.get("translated_sentence", "")
          }
          writer.writerow(filtered_item)
    
    print(f"Saved CSV results to {csv_filename}")
    
    return json_filename, csv_filename

def find_latest_checkpoint(checkpoint_dir, model_name=None):
    """
    Finds the latest checkpoint to resume from
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        model_name: Optional model name to filter checkpoints
        
    Returns:
        Tuple of (run_id, start_index) or (None, 0) if no checkpoint found
    """
    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    # First check if there's a latest_completed pointer
    latest_file = os.path.join(checkpoint_dir, "latest_completed.txt")
    if os.path.exists(latest_file):
        with open(latest_file, 'r') as f:
            latest_run_id = f.read().strip()
            
        # Find the corresponding checkpoint info
        info_file = os.path.join(checkpoint_dir, f"checkpoint_info_{latest_run_id}.json")
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                info = json.load(f)
                
            # If filtering by model name and this doesn't match, ignore it
            if model_name and info.get("model") != model_name:
                pass
            else:
                # Check if this run was completed
                if info.get("status") == "completed":
                    print(f"Found previously completed run: {latest_run_id}")
                    print(f"To start a new run, use --resume=new")
                    return latest_run_id, info.get("current_index", 0)
    
    # Look for checkpoint info files
    info_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_info_")]
    
    if not info_files:
        return None, 0
        
    # Load all checkpoint infos
    checkpoints = []
    for info_file in info_files:
        try:
            with open(os.path.join(checkpoint_dir, info_file), 'r') as f:
                info = json.load(f)
                
            # If filtering by model name and this doesn't match, skip
            if model_name and info.get("model") != model_name:
                continue
                
            checkpoints.append(info)
        except:
            continue
            
    if not checkpoints:
        return None, 0
        
    # Find the most recent one that's in progress
    in_progress = [c for c in checkpoints if c.get("status") == "in_progress"]
    if in_progress:
        # Sort by last_update
        latest = sorted(in_progress, key=lambda x: x.get("last_update", ""), reverse=True)[0]
        run_id = latest.get("run_id")
        current_index = latest.get("current_index", 0)
        
        print(f"Found in-progress checkpoint: {run_id}")
        print(f"Progress: {current_index}/{latest.get('total_sentences', '?')} sentences ({latest.get('progress_percentage', '?')}%)")
        
        return run_id, current_index
    
    return None, 0

def load_results_from_checkpoint(checkpoint_dir, run_id):
    """
    Loads translation results from a checkpoint
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        run_id: Run ID to load
        
    Returns:
        List of translation results or empty list if not found
    """
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_{run_id}.json")
    
    if not os.path.exists(checkpoint_file):
        return []
        
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            return checkpoint_data.get("translations", [])
    except:
        print(f"Error loading checkpoint file: {checkpoint_file}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Idiom Translation using vLLM")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Model name or path")
    parser.add_argument("--csv", type=str, default="idiom_data/idiom_dataset_en.csv",
                        help="Path to CSV file with idiom sentences")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens to generate")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                        help="Save checkpoint after processing this many sentences")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default="auto",
                        choices=["auto", "new", "latest"],
                        help="Resume behavior: 'auto' to find and resume from latest checkpoint, 'new' to start fresh, 'latest' to force resume from latest")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Specific run ID to resume from (overrides --resume)")
    parser.add_argument("--start-index", type=int, default=None,
                        help="Start processing from this index (0-based)")
    
    args = parser.parse_args()
    
    # Determine if we should resume from a checkpoint
    run_id = None
    start_idx = 0
    results = []
    
    # If specific run ID provided, use that
    if args.run_id:
        run_id = args.run_id
        results = load_results_from_checkpoint(args.checkpoint_dir, run_id)
        if results:
            start_idx = len(results)
            print(f"Resuming from specific run ID: {run_id} (already processed {start_idx} sentences)")
        else:
            print(f"Could not find checkpoint for run ID: {run_id}")
            run_id = None
            
    # Otherwise check resume option
    elif args.resume != "new":
        # Find the latest checkpoint
        found_run_id, found_idx = find_latest_checkpoint(args.checkpoint_dir, args.model if args.resume == "auto" else None)
        if found_run_id:
            run_id = found_run_id
            results = load_results_from_checkpoint(args.checkpoint_dir, run_id)
            if results:
                start_idx = len(results)
                print(f"Resuming from run ID: {run_id} (already processed {start_idx} sentences)")
            else:
                print(f"Could not load results from checkpoint, starting fresh")
                run_id = None
                start_idx = 0
    
    # Override start index if specified
    if args.start_index is not None:
        start_idx = args.start_index
        print(f"Starting from specified index: {start_idx}")
    
    print("\n" + "="*60)
    print(f"Starting translation process for {args.csv}")
    print(f"Model: {args.model}")
    print(f"Checkpoint interval: Every {args.checkpoint_interval} sentences")
    print(f"Resume mode: {args.resume}")
    if run_id:
        print(f"Run ID: {run_id} (resuming from {start_idx} sentences)")
    print("="*60 + "\n")
    
    # Run the translation process
    results, run_id = translate_idioms(
        model_name=args.model,
        csv_path=args.csv,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Save the results
    if results:
        json_file, csv_file = save_results(results, args.model, run_id)
        print(f"\nProcessing complete! Results saved to:")
        print(f"  - JSON: {json_file}")
        print(f"  - CSV: {csv_file}")
    else:
        print("No results were generated.")

if __name__ == "__main__":
    main()