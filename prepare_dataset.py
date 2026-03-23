"""
dataset_preprocess.py

Extracts dialogue text from the raw dataset and saves it
as a clean training file for the model.
"""

import os

# Input file containing raw movie dialogue data.
# Output file where cleaned text will be stored.
# Paths can be adjusted depending on dataset location.
INPUT_FILE = "archive/movie_lines.txt"
OUTPUT_FILE = "data/training_text.txt"

# List to store extracted lines.
# Each entry will be a cleaned dialogue string.
# This will be written to the output file.
lines = []

# Read raw dataset file.
# Each line is split using the dataset delimiter.
# Only valid entries are processed.
with open(INPUT_FILE, encoding="latin-1") as f:
    for line in f:
        parts = line.split(" +++$+++ ")

        # Ensure line has expected format.
        # The text content is located at index 4.
        # Skip malformed or incomplete lines.
        if len(parts) == 5:
            text = parts[4].strip()

            # Only keep non-empty lines.
            # Removes blank or invalid entries.
            # Helps keep dataset clean.
            if text:
                lines.append(text)

# Print number of extracted lines.
# Useful for verifying dataset size.
# Helps confirm preprocessing worked correctly.
print("Lines extracted:", len(lines))

# Create output directory if it does not exist.
# Prevents errors when writing the file.
# exist_ok=True avoids issues if folder already exists.
os.makedirs("data", exist_ok=True)

# Write cleaned lines to output file.
# Each line is written as a separate entry.
# UTF-8 encoding ensures compatibility.
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

# Confirm output file location.
# Helps verify where dataset was saved.
# Useful for debugging and workflow tracking.
print("Dataset saved to:", OUTPUT_FILE)