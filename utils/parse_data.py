import re
import csv
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set

class IdiomDataParser:
    def __init__(self, filepath: str):
        """
        Initialize the parser with the path to the dataset.
        
        Args:
            filepath: Path to the CSV file containing the idiom data
        """
        self.filepath = filepath
        self.data = defaultdict(dict)
        self.id_to_idiom = {}
        self.idioms_set = set()
        
    def parse_data(self) -> Dict:
        """
        Parse the data from the CSV file and organize it into a structured format.
        
        Returns:
            Dict: A dictionary containing the parsed data organized by group IDs
        """
        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip the header
            
            for row in reader:
                if len(row) < 2:
                    continue  # Skip incomplete rows
                
                group_id, sentence = row[0], row[1]
                group_id = str(group_id).strip()
                
                # Extract the base ID (without the last digit)
                base_id = group_id[:-1] if len(group_id) > 1 else group_id
                
                # Initialize the group entry if it's a new group
                if base_id not in self.data:
                    self.data[base_id] = {
                        "idiom_sentence": None,
                        "good_paraphrase": None,
                        "bad_paraphrase": None,
                        "idiom": None
                    }
                
                # Extract the idiom if present
                idiom_match = re.search(r'ID(.*?)ID', sentence)
                if idiom_match:
                    idiom = idiom_match.group(1)
                    self.data[base_id]["idiom"] = idiom
                    self.data[base_id]["idiom_sentence"] = sentence
                    self.idioms_set.add(idiom)
                    self.id_to_idiom[base_id] = idiom
                elif self.data[base_id]["good_paraphrase"] is None:
                    # Assume the first non-idiom sentence is a good paraphrase
                    self.data[base_id]["good_paraphrase"] = sentence
                else:
                    # Assume any additional sentences are bad paraphrases
                    self.data[base_id]["bad_paraphrase"] = sentence
        
        return dict(self.data)
    
    def create_structured_dataset(self) -> List[Dict]:
        """
        Convert the parsed data into a structured format.
        
        Returns:
            List[Dict]: A list of dictionaries containing idiom examples
        """
        if not self.data:
            self.parse_data()
            
        structured_data = []
        
        for base_id, entry in self.data.items():
            # Skip incomplete entries
            if not (entry["idiom_sentence"] and entry["good_paraphrase"]):
                continue
                
            # Extract the idiom text and its context
            idiom = entry["idiom"]
            idiom_sentence = entry["idiom_sentence"].replace(f"ID{idiom}ID", idiom)
            good_paraphrase = entry["good_paraphrase"]
            bad_paraphrase = entry["bad_paraphrase"]
            
            data_point = {
                "id": base_id,
                "idiom": idiom,
                "idiom_sentence": idiom_sentence,
                "original_idiom_sentence": entry["idiom_sentence"],
                "good_paraphrase": good_paraphrase,
                "has_bad_paraphrase": bad_paraphrase is not None,
                "bad_paraphrase": bad_paraphrase
            }
            
            structured_data.append(data_point)
        
        return structured_data
    
    def save_json(self, output_path: str) -> None:
        """
        Save the parsed data as a JSON file.
        
        Args:
            output_path: Path where to save the JSON file
        """
        structured_data = self.create_structured_dataset()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2)
    
    def save_csv(self, output_path: str) -> None:
        """
        Save the parsed data as a CSV file.
        
        Args:
            output_path: Path where to save the CSV file
        """
        structured_data = self.create_structured_dataset()
        
        # Define the fieldnames based on the data structure
        fieldnames = [
            "id", "idiom", "idiom_sentence", "original_idiom_sentence", 
            "good_paraphrase", "has_bad_paraphrase", "bad_paraphrase"
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
        idioms_list = self.get_idioms_list()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for idiom in idioms_list:
                f.write(f"{idiom}\n")
            
    def get_idioms_list(self) -> List[str]:
        """
        Get a list of all unique idioms in the dataset.
        
        Returns:
            List[str]: A list of all unique idioms
        """
        if not self.idioms_set and self.filepath:
            self.parse_data()
        
        return list(self.idioms_set)

    def get_stats(self) -> Dict:
        """
        Get statistics about the parsed data.
        
        Returns:
            Dict: A dictionary containing various statistics
        """
        data = self.create_structured_dataset()
        
        stats = {
            "total_examples": len(data),
            "unique_idioms": len(self.get_idioms_list()),
            "examples_with_bad_paraphrase": sum(1 for item in data if item["has_bad_paraphrase"]),
            "examples_without_bad_paraphrase": sum(1 for item in data if not item["has_bad_paraphrase"])
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    import os
    
    # Create output directory if it doesn't exist
    output_dir = "idiom_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the parser with the path to your data
    input_file = "data/best_data_trainer.csv"
    parser = IdiomDataParser(input_file)
    
    # Parse the data
    parsed_data = parser.parse_data()
    
    # Create a structured format for the data
    structured_data = parser.create_structured_dataset()
    
    # Print some statistics
    stats = parser.get_stats()
    print(f"Processing file: {input_file}")
    print(f"Total number of idiom examples: {stats['total_examples']}")
    print(f"Total number of unique idioms: {stats['unique_idioms']}")
    print(f"Examples with bad paraphrases: {stats['examples_with_bad_paraphrase']}")
    print(f"Examples without bad paraphrases: {stats['examples_without_bad_paraphrase']}")
    
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