import csv
import os


def load_csv(file_path):
    """Load data from a CSV file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data


def save_csv(file_path, data, fieldnames):
    """Save data to a CSV file."""
    with open(file_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


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
    # Get the CSV file path
    csv_path = input("Enter the path to your CSV file: ")

    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    # Load the CSV data
    try:
        data = load_csv(csv_path)
        print(f"Loaded {len(data)} idiom entries.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Check required columns
    required_columns = ['idiom_sentence', 'good_paraphrase', 'idiom']
    if not all(column in data[0] for column in required_columns):
        print(f"CSV file is missing required columns. Needed: {required_columns}")
        print(f"Found: {list(data[0].keys())}")
        return

    # Output file path
    output_path = input("Enter the path for the output CSV file (leave blank to overwrite original): ")
    if not output_path:
        output_path = csv_path

    # Track progress
    progress_file = "idiom_rating_progress.txt"
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

    # Get field names (column headers) from the first row
    fieldnames = list(data[0].keys())
    if 'idiom_sentence_rating' not in fieldnames:
        fieldnames.append('idiom_sentence_rating')

    # Process each entry
    for i, entry in enumerate(data[start_index:], start=start_index):
        print("\n" + "=" * 70)
        print(f"Entry {i + 1} of {len(data)}")

        if 'id' in entry:
            print(f"ID: {entry['id']}")

        print(f"Idiom: {entry['idiom']}")

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
    save_csv(output_path, data, fieldnames)
    print(f"\nRating complete! Updated dataset saved to {output_path}")

    # Clean up progress file if completed
    if i == len(data) - 1:
        if os.path.exists(progress_file):
            os.remove(progress_file)


if __name__ == "__main__":
    main()
