import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from memory import Memory

def test_memory_add_and_trim():
    mem = Memory()

    for i in range(20):
        mem.add_user_message(f"msg {i}")

    history = mem.get_history()

    assert len(history) <= mem.max_messages
    assert history[-1]["content"] == "msg 19"