import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tokenizer import Tokenizer

def test_tokenizer_encode_decode():
    tokenizer = Tokenizer()

    text = "Hello world hello"
    tokenizer.build_vocab(text)

    tokens = tokenizer.encode(text)

    assert isinstance(tokens, list)
    assert len(tokens) > 0

    decoded = tokenizer.decode(tokens)

    assert isinstance(decoded, str)
    assert "hello" in decoded.lower()