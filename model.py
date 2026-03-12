"""
model.py
--------

Defines the MiniGPT language model used for training.
"""

import torch
import torch.nn as nn


class MiniGPT(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Recurrent layer
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):

        # Convert tokens to embeddings
        x = self.embedding(x)

        # Run through GRU
        output, _ = self.rnn(x)

        # Convert to vocabulary predictions
        logits = self.fc(output)

        return logits