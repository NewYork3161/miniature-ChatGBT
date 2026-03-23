"""
dataset_loader.py

Word-based dataset loader.

This module prepares text data for training by converting it into
token sequences. It works with PyTorch's Dataset class to provide
input-target pairs for model training.
"""

import torch
from torch.utils.data import Dataset


class DatasetLoader(Dataset):
    """
    Custom dataset for loading and processing text data.

    This class converts raw text into token sequences and formats
    them into input-output pairs. It is designed to be used with
    PyTorch DataLoader for batching during training.
    """

    def __init__(self, file_path, tokenizer, seq_length=32):
        """
        Load and preprocess the dataset.

        Reads the text file, builds the tokenizer vocabulary,
        and converts the text into a tensor of token IDs.
        """

        # Read the entire dataset file as a single string.
        # The file is expected to contain raw text data.
        # UTF-8 encoding ensures compatibility with most text sources.
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Build vocabulary from the dataset text.
        # This maps words or tokens to unique integer IDs.
        # The tokenizer uses this mapping for encoding and decoding.
        tokenizer.build_vocab(text)

        # Convert the text into a sequence of token IDs.
        # Each token represents a word or unit of text.
        # The result is a list of integers.
        tokens = tokenizer.encode(text)

        # Store tokens as a PyTorch tensor.
        # This allows efficient slicing and compatibility with training.
        # dtype long is required for embedding layers.
        self.data = torch.tensor(tokens, dtype=torch.long)

        # Store the sequence length used for training samples.
        # This defines how many tokens are included in each input.
        # It also determines how targets are generated.
        self.seq_length = seq_length

    def __len__(self):
        """
        Return the number of available training samples.

        The dataset length is reduced by seq_length because each
        sample requires a full sequence plus one additional token
        for the target shift.
        """
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        """
        Retrieve a single training sample.

        Returns:
        x: input sequence of length seq_length
        y: target sequence shifted by one position
        """

        # Input sequence from current index.
        # This is the data the model will use as input.
        # It contains seq_length tokens.
        x = self.data[idx:idx + self.seq_length]

        # Target sequence shifted by one position.
        # Each token in y is the next token after x.
        # This trains the model to predict the next word.
        y = self.data[idx + 1:idx + self.seq_length + 1]

        # Return input-target pair.
        # This format is standard for sequence prediction tasks.
        return x, y