import sys
import os

# Add the parent directory of this file to the Python module search path.
# This allows importing modules that are located one directory above this file.
# os.path.dirname(__file__) gets the directory of the current file.
# os.path.join(..., "..") moves one level up.
# os.path.abspath(...) converts the path to an absolute path.
# sys.path.append(...) adds that path so Python can locate modules there.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the main AIChatEngine class that is being tested.
from ai_chat_engine import AIChatEngine


class MockInference:
    """
    This class is a lightweight replacement for the real inference engine.

    Purpose:
    The real inference system likely loads machine learning models,
    which are slow and resource-intensive. This mock version avoids
    that overhead and provides a predictable output for testing.

    Behavior:
    The generate method ignores all inputs and always returns a fixed string.
    This allows the test to verify that the AIChatEngine correctly calls
    the inference layer without depending on actual model execution.
    """

    def __init__(self):
        # No initialization is required because this is a mock object.
        pass

    def generate(self, user_input, history, context):
        """
        Simulates generating a response from an AI model.

        Parameters:
        user_input: The current user message.
        history: The conversation history stored in memory.
        context: Additional contextual data, such as search results.

        Returns:
        A fixed string "mock response" to simulate model output.
        """
        return "mock response"


class MockSearch:
    """
    This class is a mock replacement for a search or retrieval system.

    Purpose:
    The real search system may query databases, APIs, or external services.
    This mock avoids those dependencies and ensures consistent test behavior.

    Behavior:
    The search method always returns a fixed string regardless of input.
    """

    def search(self, query):
        """
        Simulates performing a search based on a query.

        Parameters:
        query: The search query string.

        Returns:
        A fixed string representing mock search data.
        """
        return "mock search data"


def test_ai_engine_response():
    """
    This function tests the generate_response method of the AIChatEngine.

    Key idea:
    Instead of calling the normal constructor, which may initialize
    heavy resources such as machine learning models, the object is
    created using __new__ to bypass initialization.

    Steps performed:
    1. Create an AIChatEngine instance without running __init__.
    2. Manually inject dependencies (memory, inference, search).
    3. Call generate_response with a sample input.
    4. Verify that the response matches the expected mock output.
    5. Verify that conversation history is updated correctly.
    """

    # Create an instance of AIChatEngine without executing its __init__ method.
    # This avoids loading heavy resources such as models or external systems.
    engine = AIChatEngine.__new__(AIChatEngine)

    # Import the Memory class used to store conversation history.
    from memory import Memory

    # Manually assign required components to the engine.
    # These replace the normal dependencies that would be set in __init__.
    engine.memory = Memory()
    engine.inference = MockInference()
    engine.search = MockSearch()

    # Call the method under test with a sample input.
    response = engine.generate_response("hello")

    # Verify that the response matches the expected output from MockInference.
    assert response == "mock response"

    # Verify that the memory history has been updated.
    # Typically, a conversation includes at least:
    # 1. The user's input
    # 2. The AI's response
    # Therefore, the history length should be at least 2 entries.
    assert len(engine.memory.get_history()) >= 2