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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer()

    dataset = DatasetLoader(Config.DATASET_PATH, tokenizer, Config.SEQ_LENGTH)

    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    vocab = tokenizer.word2idx
    vocab_size = len(vocab)

    print("Vocab size:", vocab_size)

    model = MiniGPT(vocab_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(Config.EPOCHS):

        total_loss = 0

        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")

        for x, y in dataloader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x)

            loss = criterion(
                output.view(-1, vocab_size),
                y.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Loss: {total_loss / len(dataloader):.4f}")

    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), Config.MODEL_PATH)

    with open("models/vocab.json", "w") as f:
        json.dump(vocab, f)

    print("Training complete!")


if __name__ == "__main__":
    main()