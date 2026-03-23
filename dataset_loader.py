"""
dataset_loader.py
Word-based dataset loader
"""

import torch
from torch.utils.data import Dataset


class DatasetLoader(Dataset):

    def __init__(self, file_path, tokenizer, seq_length=32):

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokenizer.build_vocab(text)

        tokens = tokenizer.encode(text)

        self.data = torch.tensor(tokens, dtype=torch.long)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):

        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+1:idx+self.seq_length+1]

        return x, y