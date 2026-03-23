import torch
import json
import os

from model import MiniGPT
from tokenizer import Tokenizer
from config import Config


class InferenceEngine:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = Tokenizer()

        with open("models/vocab.json", "r") as f:
            vocab = json.load(f)

        self.tokenizer.word2idx = vocab
        self.tokenizer.idx2word = {v: k for k, v in vocab.items()}

        self.model = MiniGPT(len(vocab)).to(self.device)

        self.model.load_state_dict(
            torch.load(Config.MODEL_PATH, map_location=self.device)
        )

        self.model.eval()

    def generate(self, user_input, history=None, context=""):

        input_tokens = self.tokenizer.encode(user_input)

        generated = input_tokens.copy()

        for _ in range(30):

            x = torch.tensor([generated[-32:]], dtype=torch.long).to(self.device)

            with torch.no_grad():
                output = self.model(x)

            next_token = torch.argmax(output[0, -1]).item()

            generated.append(next_token)

        response = self.tokenizer.decode(generated[len(input_tokens):])

        return response