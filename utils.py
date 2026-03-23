"""
utils.py

Utility helper functions used across the chatbot system.

Provides common operations such as logging, file handling,
timestamps, and basic text processing.
"""

import os
import datetime
from config import Config


# -------------------------------------------------
# Logging
# -------------------------------------------------

def log(message):
    """
    Print debug messages when debug mode is enabled.

    Used for development and troubleshooting without
    affecting production output.
    """

    # Only print when debug mode is active.
    # Prevents unnecessary console output in production.
    # Keeps logging controlled through config.
    if Config.DEBUG_MODE:
        print(f"[DEBUG] {message}")


# -------------------------------------------------
# Timestamp
# -------------------------------------------------

def get_timestamp():
    """
    Return the current timestamp as a formatted string.

    Useful for logging events or tracking actions over time.
    """

    # Get current date and time.
    # Format into readable string.
    # Standard format used across the system.
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -------------------------------------------------
# Save text to file
# -------------------------------------------------

def save_text(file_path, text):
    """
    Save text to a file, overwriting existing content.

    Used for writing logs, outputs, or processed data.
    """

    # Open file in write mode.
    # Overwrites file if it already exists.
    # Uses UTF-8 encoding for compatibility.
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


# -------------------------------------------------
# Append text to file
# -------------------------------------------------

def append_text(file_path, text):
    """
    Append text to a file.

    Keeps existing content and adds new data at the end.
    """

    # Open file in append mode.
    # Adds new line after text.
    # Useful for logs or incremental writes.
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# -------------------------------------------------
# Load text file
# -------------------------------------------------

def load_text(file_path):
    """
    Load and return text from a file.

    Returns empty string if file does not exist.
    """

    # Check if file exists.
    # Prevents errors when reading missing files.
    # Returns empty string if not found.
    if not os.path.exists(file_path):
        return ""

    # Open file in read mode.
    # Read entire content into memory.
    # Return as string.
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# -------------------------------------------------
# Ensure folder exists
# -------------------------------------------------

def ensure_folder(path):
    """
    Create a folder if it does not exist.

    Helps avoid errors when saving files to new directories.
    """

    # Check if folder exists.
    # Create it if missing.
    # Ensures safe file operations.
    if not os.path.exists(path):
        os.makedirs(path)


# -------------------------------------------------
# Clean text
# -------------------------------------------------

def clean_text(text):
    """
    Perform basic text cleaning.

    Removes extra whitespace and normalizes line breaks.
    """

    # Remove leading and trailing whitespace.
    # Replace newline characters with spaces.
    # Returns cleaned string.
    text = text.strip()
    text = text.replace("\n", " ")

    return text