import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tokenizer import Tokenizer
from dataset_loader import DatasetLoader

def test_dataset_loader_basic(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("hello world hello")

    tokenizer = Tokenizer()

    dataset = DatasetLoader(str(file), tokenizer, seq_length=2)

    x, y = dataset[0]

    assert len(x) == 2
    assert len(y) == 2