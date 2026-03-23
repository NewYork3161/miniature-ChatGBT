import sys
import os

# Add the parent directory to the Python path so local modules can be imported.
# This allows access to the Memory class when running tests.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the component being tested.
from memory import Memory


def test_memory_add_and_trim():
    """
    Test that the Memory class correctly stores messages and enforces
    its maximum history size by trimming older entries.
    """

    # Create a new Memory instance.
    mem = Memory()

    # Add more messages than the allowed maximum.
    # This should trigger internal trimming of older messages.
    for i in range(20):
        mem.add_user_message(f"msg {i}")

    # Retrieve the stored message history.
    history = mem.get_history()

    # Verify that the total number of stored messages does not exceed the limit.
    assert len(history) <= mem.max_messages

    # Verify that the most recent message is preserved after trimming.
    # The last message added should always remain in history.
    assert history[-1]["content"] == "msg 19"