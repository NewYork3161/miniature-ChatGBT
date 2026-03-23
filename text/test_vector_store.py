import sys
import os

# Add the parent directory to the Python path so local modules can be imported.
# This allows access to the Tokenizer class when running tests.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the component being tested.
from tokenizer import Tokenizer


def test_tokenizer_encode_decode():
    """
    Test that the Tokenizer can build a vocabulary, encode text into tokens,
    and decode tokens back into a string.
    """

    # Create a tokenizer instance.
    tokenizer = Tokenizer()

    # Define sample input text.
    text = "Hello world hello"

    # Build the vocabulary based on the input text.
    # This step prepares internal mappings for encoding and decoding.
    tokenizer.build_vocab(text)

    # Encode the text into a list of tokens.
    tokens = tokenizer.encode(text)

    # Verify that encoding returns a non-empty list.
    assert isinstance(tokens, list)
    assert len(tokens) > 0

    # Decode the tokens back into text.
    decoded = tokenizer.decode(tokens)

    # Verify that decoding returns a string.
    assert isinstance(decoded, str)

    # Verify that expected content is preserved after encode/decode.
    # Comparison is done in lowercase to avoid case sensitivity issues.
    assert "hello" in decoded.lower()