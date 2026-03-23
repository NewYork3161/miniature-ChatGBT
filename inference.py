import torch
import json
import os

from model import MiniGPT
from tokenizer import Tokenizer
from config import Config


class InferenceEngine:
    """
    Handles model loading and text generation.

    This class prepares the tokenizer and model for inference.
    It converts user input into tokens, runs the model, and
    generates a response token by token.
    """

    def __init__(self):
        """
        Initialize tokenizer, model, and device.

        Loads vocabulary, reconstructs tokenizer mappings,
        and restores the trained model from disk.
        """

        # Select device (GPU if available, otherwise CPU).
        # This allows faster inference when CUDA is available.
        # The model and tensors must be moved to this device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize tokenizer instance.
        # This will be used to encode input text and decode output tokens.
        # Vocabulary mappings will be loaded from file.
        self.tokenizer = Tokenizer()

        # Load vocabulary from saved JSON file.
        # This ensures consistency between training and inference.
        # The tokenizer uses this mapping to convert words to indices.
        with open("models/vocab.json", "r") as f:
            vocab = json.load(f)

        # Assign vocabulary mappings to tokenizer.
        # word2idx maps words to integers.
        # idx2word reverses the mapping for decoding.
        self.tokenizer.word2idx = vocab
        self.tokenizer.idx2word = {v: k for k, v in vocab.items()}

        # Initialize model with vocabulary size.
        # The model structure must match what was used during training.
        # It is then moved to the selected device.
        self.model = MiniGPT(len(vocab)).to(self.device)

        # Load trained model weights from file.
        # map_location ensures compatibility with CPU or GPU.
        # This restores the model to its trained state.
        self.model.load_state_dict(
            torch.load(Config.MODEL_PATH, map_location=self.device)
        )

        # Set model to evaluation mode.
        # This disables training-specific behavior like dropout.
        # Required for consistent inference results.
        self.model.eval()

    def generate(self, user_input, history=None, context=""):
        """
        Generate a response from user input.

        Converts input text into tokens, feeds it into the model,
        and generates new tokens step by step.
        """

        # Encode user input into token IDs.
        # This converts text into a numerical format for the model.
        # The result is a list of integers.
        input_tokens = self.tokenizer.encode(user_input)

        # Initialize generated sequence with input tokens.
        # The model will continue generating from this starting point.
        # New tokens will be appended to this list.
        generated = input_tokens.copy()

        # Generate tokens one at a time.
        # The loop controls how long the response will be.
        # Here it generates up to 30 new tokens.
        for _ in range(30):

            # Take the last 32 tokens as input context.
            # This acts as a sliding window for the model.
            # It ensures the model only processes recent tokens.
            x = torch.tensor([generated[-32:]], dtype=torch.long).to(self.device)

            # Run the model without tracking gradients.
            # This improves performance and reduces memory usage.
            # Only forward pass is needed during inference.
            with torch.no_grad():
                output = self.model(x)

            # Select the most likely next token.
            # argmax picks the highest probability output.
            # This results in deterministic generation.
            next_token = torch.argmax(output[0, -1]).item()

            # Append the new token to the sequence.
            # This allows the next step to use updated context.
            generated.append(next_token)

        # Decode only the newly generated tokens.
        # This excludes the original input tokens.
        # The result is the final response string.
        response = self.tokenizer.decode(generated[len(input_tokens):])

        return response