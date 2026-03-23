"""
model.py

Defines the MiniGPT model.

Implements a transformer-based architecture for
sequence modeling and text generation.
"""

import torch
import torch.nn as nn


class MiniGPT(nn.Module):
    """
    Transformer-based language model.

    Uses token embeddings, positional embeddings,
    and a stack of transformer encoder layers
    to generate token predictions.
    """

    def __init__(
        self,
        vocab_size,
        embed_size=256,
        num_heads=4,
        num_layers=4,
        max_seq_length=512
    ):
        """
        Initialize model layers and architecture.

        Sets up embeddings, transformer layers,
        and output projection for token prediction.
        """
        super().__init__()

        # Store embedding size for reference.
        # Used across multiple layers.
        # Keeps model dimensions consistent.
        self.embed_size = embed_size

        # Token embedding layer.
        # Converts token IDs into dense vectors.
        # These vectors represent semantic meaning.
        self.token_embedding = nn.Embedding(vocab_size, embed_size)

        # Positional embedding layer.
        # Adds position information to each token.
        # Required because transformers do not track order by default.
        self.position_embedding = nn.Embedding(max_seq_length, embed_size)

        # Define a single transformer encoder layer.
        # batch_first=True ensures input shape is (batch, seq, features).
        # This matches how data is passed in this project.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            batch_first=True
        )

        # Stack multiple transformer layers.
        # This increases model depth and learning capacity.
        # Each layer processes the sequence representation further.
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection layer.
        # Maps transformer outputs to vocabulary size.
        # Produces logits for each token position.
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        """
        Forward pass through the model.

        Takes a sequence of token IDs and returns
        logits representing next-token predictions.
        """

        # Extract batch size and sequence length.
        # Input shape is (batch_size, seq_length).
        # Used to generate positional indices.
        batch_size, seq_length = x.shape

        # Create position indices for each token.
        # Expands to match batch size.
        # Ensures each token has a position embedding.
        positions = torch.arange(0, seq_length, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_length)

        # Generate token and positional embeddings.
        # These are combined to include both meaning and order.
        # Resulting tensor is passed into the transformer.
        token_embed = self.token_embedding(x)
        position_embed = self.position_embedding(positions)

        x = token_embed + position_embed

        # Pass embeddings through transformer layers.
        # Each layer applies attention and transformations.
        # Output maintains sequence structure.
        x = self.transformer(x)

        # Project outputs to vocabulary logits.
        # Each position predicts probability over tokens.
        # Used for next-token generation.
        logits = self.fc_out(x)

        return logits