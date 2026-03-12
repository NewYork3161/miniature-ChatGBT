"""
utils.py
---------------------------------
Utility helper functions used across MiniChatGPT.
"""

import os
import datetime
from config import Config


# -------------------------------------------------
# Logging
# -------------------------------------------------

def log(message):
    """
    Print debug messages if debug mode is enabled
    """

    if Config.DEBUG_MODE:
        print(f"[DEBUG] {message}")


# -------------------------------------------------
# Timestamp
# -------------------------------------------------

def get_timestamp():
    """
    Return a formatted timestamp string
    """

    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# -------------------------------------------------
# Save text to file
# -------------------------------------------------

def save_text(file_path, text):
    """
    Save text to a file
    """

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)


# -------------------------------------------------
# Append text to file
# -------------------------------------------------

def append_text(file_path, text):
    """
    Append text to a file
    """

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# -------------------------------------------------
# Load text file
# -------------------------------------------------

def load_text(file_path):
    """
    Load text from a file
    """

    if not os.path.exists(file_path):
        return ""

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# -------------------------------------------------
# Ensure folder exists
# -------------------------------------------------

def ensure_folder(path):
    """
    Create folder if it doesn't exist
    """

    if not os.path.exists(path):
        os.makedirs(path)


# -------------------------------------------------
# Clean text
# -------------------------------------------------

def clean_text(text):
    """
    Basic text cleaning
    """

    text = text.strip()
    text = text.replace("\n", " ")

    return text