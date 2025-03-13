import re
import csv
import json
from collections import defaultdict
from typing import List, Dict, Optional, Set

class SimpleIdiomParser:
    def __init__(self, filepath: str):
        """
        Initialize the parser with the path to the dataset.
        
        Args:
            filepath: Path to the CSV file containing the idiom data
        """
        self.filepath = filepath
        self.data = defaultdict(list)
        self.idioms_set = set()
        
    def parse_data(self) -> List[Dict]:
        """
        Parse the data from the CSV file and organize it into a structured format.
        Assumes entries with the same ID are grouped as:
        1. First entry: the idiom expression
        2. Second entry: the good paraphrase
        
        Returns:
            List[Dict]: A list of structured idiom entries
        """
        # Group entries by their base IDs
        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip the header
            
            for row in reader:
                if len(row) < 2:
                    continue  # Skip incomplete rows
                
                label, sentence = row[0], row[1]
                # Strip the last digit for grouping if needed
                base_id = label[:-1] if len(label) > 1 else label
                
                self.data[base_id].append((label, sentence))
        
        # Process each group to extract idiom and paraphrase
        structured_data = []
        for base_id, entries in self.data.items():
            # Sort entries by label to ensure consistent processing
            entries.sort(key=lambda x: x[0])
            
            # First entry should contain the idiom
            idiom_entry = None
            for label, sentence in entries:
                idiom_match = re.search(r'ID(.*?)ID', sentence)
                if idiom_match:
                    idiom = idiom_match.group(1)
                    idiom_entry = (label, sentence, idiom)
                    self.idioms_set.add(idiom)
                    break
            
            if not idiom_entry:
                continue  # Skip if no idiom found
            
            # Find the good paraphrase (entry without ID...ID)
            good_paraphrase = None
            for label, sentence in entries:
                if "ID" not in sentence:
                    good_paraphrase = (label, sentence)
                    break
            
            # Only add if we have both idiom and good paraphrase
            if idiom_entry and good_paraphrase:
                _, idiom_sentence, idiom = idiom_entry
                _, paraphrase_sentence = good_paraphrase
                
                example = {
                    "id": base_id,
                    "idiom": idiom,
                    "idiom_sentence": idiom_sentence.replace(f"ID{idiom}ID", idiom),
                    "original_idiom_sentence": idiom_sentence,
                    "good_paraphrase": paraphrase_sentence
                }
                
                structured_data.append(example)
        
        return structured_data
    
    def save_json(self, output_path: str) -> None:
        """
        Save the parsed data as a JSON file.
        
        Args:
            output_path: Path where to save the JSON file
        """
        structured_data = self.parse_data()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2)
    
    def save_csv(self, output_path: str) -> None:
        """
        Save the parsed data as a CSV file.
        
        Args:
            output_path: Path where to save the CSV file
        """
        structured_data = self.parse_data()
        
        # Define the fieldnames based on the data structure
        fieldnames = [
            "id", "idiom", "idiom_sentence", "original_idiom_sentence", 
            "good_paraphrase"
        ]
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in structured_data:
                writer.writerow(item)
    
    def save_idioms_list(self, output_path: str) -> None:
        """
        Save the list of unique idioms to a text file.
        
        Args:
            output_path: Path where to save the idioms list
        """
        parsed_data = self.parse_data()  # Make sure idioms are collected
        idioms_list = list(self.idioms_set)
        idioms_list.sort()  # Sort alphabetically
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for idiom in idioms_list:
                f.write(f"{idiom}\n")
            
    def get_stats(self) -> Dict:
        """
        Get statistics about the parsed data.
        
        Returns:
            Dict: A dictionary containing various statistics
        """
        data = self.parse_data()
        
        stats = {
            "total_examples": len(data),
            "unique_idioms": len(self.idioms_set)
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    import os
    import sys
    
    # Create output directory if it doesn't exist
    output_dir = "idiom_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get input file from command line or use default
    input_file = sys.argv[1] if len(sys.argv) > 1 else "idiom_data.csv"
    
    # Initialize the parser with the path to your data
    parser = SimpleIdiomParser(input_file)
    
    # Parse the data
    structured_data = parser.parse_data()
    
    # Print some statistics
    stats = parser.get_stats()
    print(f"Processing file: {input_file}")
    print(f"Total number of idiom examples: {stats['total_examples']}")
    print(f"Total number of unique idioms: {stats['unique_idioms']}")
    
    # Save the data in different formats
    parser.save_json(os.path.join(output_dir, "idiom_dataset.json"))
    parser.save_csv(os.path.join(output_dir, "idiom_dataset.csv"))
    parser.save_idioms_list(os.path.join(output_dir, "unique_idioms.txt"))
    
    print(f"\nFiles saved in '{output_dir}' directory:")
    print(f"1. idiom_dataset.json - Complete structured dataset in JSON format")
    print(f"2. idiom_dataset.csv - Complete structured dataset in CSV format")
    print(f"3. unique_idioms.txt - List of all unique idioms found in the dataset")
    
    # Print a sample entry
    if structured_data:
        print("\nSample entry:")
        sample = structured_data[0]
        for key, value in sample.items():
            print(f"{key}: {value}")