import os

INPUT_FILE = "archive/movie_lines.txt"
OUTPUT_FILE = "data/training_text.txt"

lines = []

with open(INPUT_FILE, encoding="latin-1") as f:
    for line in f:
        parts = line.split(" +++$+++ ")
        if len(parts) == 5:
            text = parts[4].strip()
            if text:
                lines.append(text)

print("Lines extracted:", len(lines))

os.makedirs("data", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print("Dataset saved to:", OUTPUT_FILE)