import sys
import os

# Add the parent directory to the Python path so local modules can be imported.
# This allows access to tokenizer and dataset_loader when running tests.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the components being tested.
from tokenizer import Tokenizer
from dataset_loader import DatasetLoader


def test_dataset_loader_basic(tmp_path):
    """
    Test that DatasetLoader correctly processes a simple text file
    and returns input/output sequences of the expected length.

    tmp_path:
    A pytest-provided temporary directory used to create test files
    without affecting the real filesystem.
    """

    # Create a temporary text file with simple repeating content.
    # This ensures predictable tokenization and dataset behavior.
    file = tmp_path / "test.txt"
    file.write_text("hello world hello")

    # Initialize the tokenizer used to convert text into tokens.
    tokenizer = Tokenizer()

    # Create the dataset with a sequence length of 2.
    # The dataset should split the tokenized text into input/output pairs.
    dataset = DatasetLoader(str(file), tokenizer, seq_length=2)

    # Retrieve the first sample from the dataset.
    # x is the input sequence, y is the target sequence.
    x, y = dataset[0]

    # Verify that both input and target sequences match the expected length.
    # This ensures the dataset slicing logic is working correctly.
    assert len(x) == 2
    assert len(y) == 2