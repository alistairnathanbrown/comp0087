import json
import os


def load_dataset(file_path):
    """Load the JSON dataset from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_dataset(file_path, data):
    """Save the dataset back to a file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main():
    # Get the dataset file path
    dataset_path = input("Enter the path to your JSON dataset file: ")

    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"File not found: {dataset_path}")
        return

    # Load the dataset
    try:
        data = load_dataset(dataset_path)
        print(f"Loaded {len(data)} idiom entries.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Output file path
    output_path = input("Enter the path for the output JSON file (leave blank to overwrite original): ")
    if not output_path:
        output_path = dataset_path

    # Track progress
    progress_file = "ranking_progress.txt"
    start_index = 0

    # Check if progress file exists to resume
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            try:
                start_index = int(f.read().strip())
                print(f"Resuming from entry {start_index + 1}")
            except:
                print("Could not read progress file. Starting from the beginning.")

    # Define the ranking scale
    print("\nRanking Scale:")
    print("1: Very poor or incorrect usage")
    print("2: Below average, somewhat awkward")
    print("3: Average, acceptable usage")
    print("4: Good, natural usage")
    print("5: Excellent, particularly apt usage")

    # Process each idiom entry
    for i, entry in enumerate(data[start_index:], start=start_index):
        print("\n" + "=" * 50)
        print(f"Entry {i + 1} of {len(data)}")
        print(f"ID: {entry['id']}")
        print(f"Idiom: {entry['idiom']}")

        print("\nSentence with idiom:")
        print(entry['idiom_sentence'])

        # Get user ranking
        while True:
            try:
                rank = int(input("\nRank the idiom usage (1-5): "))
                if 1 <= rank <= 5:
                    break
                else:
                    print("Please enter a number between a 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")

        # Add ranking to the entry
        entry['idiom_usage_rank'] = rank

        # Save progress
        with open(progress_file, 'w') as f:
            f.write(str(i + 1))

        # Option to pause
        if i < len(data) - 1:  # Not the last entry
            cont = input("\nPress Enter to continue to the next entry, or type 'q' to quit: ")
            if cont.lower() == 'q':
                print("Progress saved. You can resume later.")
                break

    # Save the updated dataset
    save_dataset(output_path, data)
    print(f"\nRanking complete! Updated dataset saved to {output_path}")

    # Clean up progress file if completed
    if i == len(data) - 1:
        if os.path.exists(progress_file):
            os.remove(progress_file)


if __name__ == "__main__":
    main()