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


def get_rating():
    """Get user rating on a 1-4 scale."""
    while True:
        try:
            rating = int(input("\nRate the idiom sentence (1-4): "))
            if 1 <= rating <= 4:
                return rating
            else:
                print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Please enter a valid number.")


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
    progress_file = "idiom_sentence_progress.txt"
    start_index = 0

    # Check if progress file exists to resume
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            try:
                start_index = int(f.read().strip())
                print(f"Resuming from entry {start_index + 1}")
            except:
                print("Could not read progress file. Starting from the beginning.")

    # Define the rating scale
    print("\nIdiom Sentence Rating Scale:")
    print("1: Poor - Awkward or unnatural usage of the idiom")
    print("2: Below average - Somewhat forced usage of the idiom")
    print("3: Average - Acceptable usage of the idiom")
    print("4: Good - Natural and appropriate usage of the idiom")

    # Process each idiom entry sequentially by position in the dataset
    for i, entry in enumerate(data[start_index:], start=start_index):
        print("\n" + "=" * 70)
        print(f"Entry {i + 1} of {len(data)} (Sequential position)")
        print(f"Dataset ID: {entry['id']}")
        print(f"Idiom: {entry['idiom']}")

        # Check if the required fields exist
        if 'idiom_sentence' not in entry or 'good_paraphrase' not in entry:
            print("This entry is missing required fields. Skipping.")
            continue

        # Display both sentences
        print("\n1. Original idiom sentence:")
        print(entry['idiom_sentence'])

        print("\n2. Good paraphrase (for reference):")
        print(entry['good_paraphrase'])

        # Get rating
        print("\nHow would you rate the quality of the idiom usage in the original sentence?")
        idiom_rating = get_rating()

        # Add rating to the entry
        entry['idiom_sentence_rating'] = idiom_rating

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
    print(f"\nRating complete! Updated dataset saved to {output_path}")

    # Clean up progress file if completed
    if i == len(data) - 1:
        if os.path.exists(progress_file):
            os.remove(progress_file)


if __name__ == "__main__":
    main()
