import re
import csv
import json
import sys
import os
from collections import defaultdict
from typing import List, Dict

class SimpleIdiomParser:
    def __init__(self, filepath: str):
        """
        Initialize the parser with the path to the dataset.
        
        Args:
            filepath: Path to the CSV file containing the idiom data
        """
        self.filepath = filepath
        self.pairs = []
        self.idioms_set = set()
        
    def parse_data(self) -> List[Dict]:
        """
        Parse the data from the CSV file. For each unique ID, 
        treat the first occurrence as the idiomatic sentence and 
        the second as the good paraphrase.
        
        Returns:
            List[Dict]: A list of idiom-paraphrase pairs
        """
        # Group entries by their IDs
        id_sentences = defaultdict(list)
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip the header if present
            
            for row in reader:
                if len(row) < 2:
                    continue  # Skip incomplete rows
                
                id_value = row[0]
                sentence = row[1]
                id_sentences[id_value].append(sentence)
        
        # Process each ID that has exactly 2 sentences
        for id_value, sentences in id_sentences.items():
            if len(sentences) >= 2:  # Need at least 2 sentences
                idiom_sentence = sentences[0]
                paraphrase = sentences[1]
                
                # Extract the idiom from the first sentence (using ID...ID pattern)
                idiom_match = re.search(r'ID(.*?)ID', idiom_sentence)
                if idiom_match:
                    idiom = idiom_match.group(1)
                    self.idioms_set.add(idiom)
                    
                    # Replace ID...ID with just the idiom
                    clean_idiom_sentence = idiom_sentence.replace(f"ID{idiom}ID", idiom)
                    
                    pair = {
                        "id": id_value,
                        "idiom": idiom,
                        "idiom_sentence": clean_idiom_sentence,
                        "original_idiom_sentence": idiom_sentence,
                        "good_paraphrase": paraphrase
                    }
                    
                    self.pairs.append(pair)
        
        return self.pairs
    
    def save_json(self, output_path: str) -> None:
        """
        Save the paired data as a JSON file.
        
        Args:
            output_path: Path where to save the JSON file
        """
        if not self.pairs:
            self.parse_data()
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.pairs, f, indent=2)
    
    def save_csv(self, output_path: str) -> None:
        """
        Save the paired data as a CSV file.
        
        Args:
            output_path: Path where to save the CSV file
        """
        if not self.pairs:
            self.parse_data()
            
        # Define the fieldnames based on the data structure
        fieldnames = [
            "id", "idiom", "idiom_sentence", "original_idiom_sentence", 
            "good_paraphrase"
        ]
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pair in self.pairs:
                writer.writerow(pair)
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the parsed data.
        
        Returns:
            Dict: Statistics about the data
        """
        if not self.pairs:
            self.parse_data()
            
        return {
            "total_pairs": len(self.pairs),
            "unique_idioms": len(self.idioms_set)
        }


# Example usage
if __name__ == "__main__":
    # Get input file from command line or use default
    input_file = sys.argv[1] if len(sys.argv) > 1 else "idiom_data.csv"
    
    # Create output directory if it doesn't exist
    output_dir = "idiom_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found!")
        sys.exit(1)
    
    print(f"Processing file: {input_file}")
    
    # Initialize the parser with the path to your data
    parser = SimpleIdiomParser(input_file)
    
    # Parse the data
    pairs = parser.parse_data()
    
    # Print statistics
    stats = parser.get_stats()
    print(f"Total idiom-paraphrase pairs found: {stats['total_pairs']}")
    print(f"Total unique idioms: {stats['unique_idioms']}")
    
    # Save the data
    json_path = os.path.join(output_dir, "idiom_dataset.json")
    csv_path = os.path.join(output_dir, "idiom_dataset.csv")
    
    parser.save_json(json_path)
    parser.save_csv(csv_path)
    
    print(f"Files saved:")
    print(f"- JSON: {json_path}")
    print(f"- CSV: {csv_path}")
    
    # Print a sample entry if available
    if pairs:
        print("\nSample pair:")
        sample = pairs[0]
        for key, value in sample.items():
            print(f"{key}: {value}")