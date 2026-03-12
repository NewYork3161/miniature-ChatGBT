import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import time

from dataset_loader import DatasetLoader
from model import MiniGPT
from config import Config


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -----------------------
    # Dataset
    # -----------------------
    dataset = DatasetLoader(Config.DATASET_PATH, Config.SEQ_LENGTH)

    vocab_mapping = dataset.char_to_idx
    VOCAB_SIZE = len(vocab_mapping)

    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )

    print("Vocabulary size:", VOCAB_SIZE)
    print("Dataset size:", len(dataset))
    print("Batches per epoch:", len(dataloader))

    # -----------------------
    # Model
    # -----------------------
    model = MiniGPT(
        vocab_size=VOCAB_SIZE,
        embed_size=Config.EMBED_SIZE,
        hidden_size=Config.HIDDEN_SIZE
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # -----------------------
    # Training Loop
    # -----------------------
    print("\nTraining started...\n")

    for epoch in range(Config.EPOCHS):

        total_loss = 0

        print(f"Epoch {epoch+1}/{Config.EPOCHS}")

        for batch_idx, (x, y) in enumerate(dataloader):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)

            loss = criterion(
                outputs.view(-1, VOCAB_SIZE),
                y.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch+1} finished | Avg Loss {avg_loss:.4f}\n")

    # -----------------------
    # Save model
    # -----------------------
    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), Config.MODEL_PATH)

    with open("models/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_mapping, f)

    print("\nModel and Vocabulary saved successfully.")


if __name__ == "__main__":
    main()