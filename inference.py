"""
inference.py
---------------------------------
Runs the trained MiniChatGPT neural network model
and generates responses for user input.
"""

import os
import json
import torch
import torch.nn.functional as F

from model import MiniGPT
from tokenizer import Tokenizer
from config import Config


class InferenceEngine:

    def __init__(self):

        # -------------------------------------------------
        # Device setup
        # -------------------------------------------------
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # -------------------------------------------------
        # Tokenizer
        # -------------------------------------------------
        self.tokenizer = Tokenizer()

        vocab_file = "models/vocab.json"

        if os.path.exists(vocab_file) and os.path.getsize(vocab_file) > 0:

            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)

            self.tokenizer.char2idx = vocab
            self.tokenizer.idx2char = {v: k for k, v in vocab.items()}

            print(f"[Tokenizer] Loaded vocabulary with {len(vocab)} tokens.")

        else:

            print("[Error] vocab.json not found. Run train.py first.")
            self.tokenizer.char2idx = {}
            self.tokenizer.idx2char = {}

        # -------------------------------------------------
        # Vocabulary size
        # -------------------------------------------------
        self.vocab_size = len(self.tokenizer.char2idx)

        if self.vocab_size == 0:
            self.vocab_size = 65

        # -------------------------------------------------
        # Model initialization
        # -------------------------------------------------
        self.model = MiniGPT(
            vocab_size=self.vocab_size,
            embed_size=Config.EMBED_SIZE,
            hidden_size=Config.HIDDEN_SIZE
        ).to(self.device)

        model_path = Config.MODEL_PATH

        # -------------------------------------------------
        # Load model weights
        # -------------------------------------------------
        if os.path.exists(model_path):

            try:

                state_dict = torch.load(
                    model_path,
                    map_location=self.device,
                    weights_only=True
                )

                self.model.load_state_dict(state_dict)
                self.model.eval()

                print(f"[Inference] Model loaded successfully from {model_path}")

            except Exception as e:

                print(f"[Inference] Failed to load model weights: {e}")

        else:

            print(f"[Inference] Model file not found at: {model_path}")


    def generate(self, user_input, history=None, context=""):
        """
        Generate a response using the trained model
        """

        prompt = context + " " + user_input if context else user_input

        tokens = self.tokenizer.encode(prompt)

        if not tokens:
            tokens = [0]

        tokens = tokens[-Config.SEQ_LENGTH:]

        generated = tokens.copy()

        max_new_tokens = 40

        for _ in range(max_new_tokens):

            input_tensor = torch.tensor(
                [generated[-Config.SEQ_LENGTH:]],
                dtype=torch.long
            ).to(self.device)

            with torch.no_grad():

                output = self.model(input_tensor)

                probabilities = F.softmax(output[0, -1], dim=0)

                next_token = torch.multinomial(probabilities, 1).item()

            generated.append(next_token)

        response_tokens = generated[len(tokens):]

        response = self.tokenizer.decode(response_tokens)

        return response