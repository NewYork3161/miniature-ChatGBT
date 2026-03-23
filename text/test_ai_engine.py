import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_chat_engine import AIChatEngine


class MockInference:
    def __init__(self):
        pass

    def generate(self, user_input, history, context):
        return "mock response"


class MockSearch:
    def search(self, query):
        return "mock search data"


def test_ai_engine_response():

    # 🔥 create engine WITHOUT running heavy inference init
    engine = AIChatEngine.__new__(AIChatEngine)

    # manually inject dependencies
    from memory import Memory
    engine.memory = Memory()
    engine.inference = MockInference()
    engine.search = MockSearch()

    response = engine.generate_response("hello")

    assert response == "mock response"
    assert len(engine.memory.get_history()) >= 2