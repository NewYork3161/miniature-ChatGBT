"""
model.py
Transformer-based MiniGPT (clean + optimized)
"""

import torch
import torch.nn as nn


class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_size=256,
        num_heads=4,
        num_layers=4,
        max_seq_length=512
    ):
        super().__init__()

        self.embed_size = embed_size

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_size)

        # Positional embedding
        self.position_embedding = nn.Embedding(max_seq_length, embed_size)

        # Transformer encoder layer (FIXED with batch_first)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            batch_first=True   # 🔥 FIXES WARNING
        )

        # Transformer stack
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):

        batch_size, seq_length = x.shape

        # Create position indices
        positions = torch.arange(0, seq_length, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_length)

        # Embeddings
        token_embed = self.token_embedding(x)
        position_embed = self.position_embedding(positions)

        x = token_embed + position_embed

        # Transformer
        x = self.transformer(x)

        # Output logits
        logits = self.fc_out(x)

        return logits