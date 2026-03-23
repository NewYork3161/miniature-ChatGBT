"""
train.py

Trains the MiniGPT model using the prepared dataset.

Handles data loading, model training, loss calculation,
and saving the trained model and vocabulary.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json

from dataset_loader import DatasetLoader
from model import MiniGPT
from tokenizer import Tokenizer
from config import Config


def main():
    """
    Main training function.

    Sets up device, loads dataset, initializes model,
    and runs the training loop for a fixed number of epochs.
    """

    # Select device (GPU if available, otherwise CPU).
    # Improves training speed when CUDA is available.
    # All tensors and model must be moved to this device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer.
    # Builds vocabulary from dataset during loading.
    # Used for encoding text into token IDs.
    tokenizer = Tokenizer()

    # Load dataset and prepare sequences.
    # DatasetLoader converts text into token sequences.
    # Sequence length is defined in config.
    dataset = DatasetLoader(Config.DATASET_PATH, tokenizer, Config.SEQ_LENGTH)

    # Create DataLoader for batching.
    # Enables efficient iteration and shuffling.
    # Batch size is controlled by config.
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Get vocabulary and size.
    # Needed to define model output dimensions.
    # Vocabulary comes from tokenizer.
    vocab = tokenizer.word2idx
    vocab_size = len(vocab)

    print("Vocab size:", vocab_size)

    # Initialize model.
    # Vocabulary size determines output layer size.
    # Model is moved to selected device.
    model = MiniGPT(vocab_size).to(device)

    # Define optimizer and loss function.
    # Adam is used for stable training.
    # CrossEntropyLoss is standard for classification tasks.
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Training loop over epochs.
    # Each epoch processes the entire dataset once.
    # Loss is tracked for monitoring training progress.
    for epoch in range(Config.EPOCHS):

        total_loss = 0

        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")

        # Iterate over batches.
        # Each batch contains input (x) and target (y).
        # Data is moved to the correct device.
        for x, y in dataloader:

            x = x.to(device)
            y = y.to(device)

            # Reset gradients.
            # Required before each backward pass.
            # Prevents gradient accumulation.
            optimizer.zero_grad()

            # Forward pass.
            # Model outputs logits for each token.
            # Shape matches input sequence.
            output = model(x)

            # Compute loss.
            # Flatten output and target for comparison.
            # Measures prediction error.
            loss = criterion(
                output.view(-1, vocab_size),
                y.view(-1)
            )

            # Backpropagation.
            # Computes gradients for all parameters.
            # Updates model weights using optimizer.
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for epoch.
        # Helps track training progress.
        # Lower loss generally indicates improvement.
        print(f"Loss: {total_loss / len(dataloader):.4f}")

    # Create models directory if needed.
    # Prevents file write errors.
    # Keeps saved files organized.
    os.makedirs("models", exist_ok=True)

    # Save trained model weights.
    # Allows reuse without retraining.
    # Path is defined in config.
    torch.save(model.state_dict(), Config.MODEL_PATH)

    # Save vocabulary for inference.
    # Ensures tokenizer consistency.
    # Stored as JSON for easy loading.
    with open("models/vocab.json", "w") as f:
        json.dump(vocab, f)

    print("Training complete!")


# Run training only if script is executed directly.
# Prevents automatic execution when imported.
# Standard Python entry pattern.
if __name__ == "__main__":
    main()