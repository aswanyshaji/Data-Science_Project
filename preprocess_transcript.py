import re
import sys

def clean_and_normalize_text(text):
    """
    Clean and normalize the transcript text by:
    1. Removing filler words.
    2. Converting text to lowercase.
    3. Removing punctuation.
    4. Correcting any common typos (if applicable).
    5. Removing extra spaces.
    
    Parameters:
    - text (str): The input transcript text.
    
    Returns:
    - str: The cleaned and normalized text.
    """
    # List of filler words to remove
    filler_words = ['um', 'uh', 'like', 'you know', 'so', 'basically', 'actually', 'literally']

    # Remove filler words
    for filler in filler_words:
        text = re.sub(r'\b' + filler + r'\b', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocess_transcript(input_file_path, output_file_path):
    """
    Read the transcript from the input file, preprocess it, and save the cleaned text to the output file.
    
    Parameters:
    - input_file_path (str): Path to the input transcript file.
    - output_file_path (str): Path to save the preprocessed transcript file.
    """
    # Read the input transcript file
    with open(input_file_path, 'r') as file:
        transcript_text = file.read()

    # Preprocess the transcript text
    cleaned_text = clean_and_normalize_text(transcript_text)

    # Save the cleaned text to the output file
    with open(output_file_path, 'w') as file:
        file.write(cleaned_text)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess_transcript.py <input_transcript_file> <output_transcript_file>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    preprocess_transcript(input_file_path, output_file_path)
    print(f"Preprocessed transcript saved to: {output_file_path}")
