"""
ai_chat_engine.py

Core controller for the chatbot system.

This module connects the main parts of the application together,
including memory, inference, and optional internet search. It handles
the flow of data from user input to final response output.
"""

# Import required subsystems
# Memory handles storing conversation history between user and AI.
# InferenceEngine generates responses using the trained model.
# InternetSearch retrieves external data when needed.
from memory import Memory
from inference import InferenceEngine
from internet_search import InternetSearch


class AIChatEngine:
    """
    Main controller for handling user input and generating responses.

    This class coordinates all subsystems and controls the response flow.
    It ensures the model receives the correct inputs and that conversation
    state is preserved across messages.
    """

    def __init__(self):
        """
        Initialize core components used by the chatbot.

        Each subsystem is created once and reused for the lifetime of the engine.
        This setup allows consistent state management and avoids repeated loading.
        """

        # Stores conversation history (user + AI messages).
        # This allows the system to maintain context across multiple turns.
        # Without this, each response would be generated independently.
        self.memory = Memory()

        # Handles model inference and response generation.
        # This is where the trained model is used to produce outputs.
        # The engine passes user input, history, and context to this component.
        self.inference = InferenceEngine()

        # Handles optional external data lookup.
        # This is used when the system detects that outside information may help.
        # Not every request uses this, so it is triggered conditionally.
        self.search = InternetSearch()

    def generate_response(self, user_input):
        """
        Generate a response for a given user message.

        This method controls the full processing pipeline from input to output.
        It combines memory, optional search, and model inference into one flow.
        """

        # Add user message to memory.
        # This ensures the latest input is included in conversation history.
        # The model will use this history when generating a response.
        self.memory.add_user_message(user_input)

        # Default to no external context.
        # Context will only be populated if search is triggered.
        # This keeps processing lightweight when external data is not needed.
        context = ""

        # Use search only if message suggests it.
        # This prevents unnecessary external calls for simple messages.
        # If triggered, search results are stored as additional context.
        if self._needs_internet_search(user_input):
            context = self.search.search(user_input)

        # Get full conversation history.
        # This provides the model with previous messages for context.
        # It helps maintain continuity across multiple turns.
        history = self.memory.get_history()

        # Generate response using the model.
        # The inference engine processes input, history, and context together.
        # The output is a generated text response.
        response = self.inference.generate(user_input, history, context)

        # Save AI response to memory.
        # This allows future messages to reference this response.
        # It keeps the conversation consistent over time.
        self.memory.add_ai_message(response)

        # Return final output.
        # The calling system will display or use this response.
        # This method only handles generation, not presentation.
        return response

    def _needs_internet_search(self, text):
        """
        Simple keyword-based check for when to use internet search.

        This method looks for patterns that suggest factual or current queries.
        It acts as a basic filter before calling the search module.
        """

        # Keywords that indicate a need for external information.
        # These are simple triggers based on common query patterns.
        # This approach is lightweight but not fully accurate.
        keywords = [
            "latest",
            "news",
            "today",
            "current",
            "recent",
            "who is",
            "what is"
        ]

        # Convert text to lowercase for consistent matching.
        # This avoids missing matches due to capitalization.
        # All comparisons are done against the lowercase version.
        text_lower = text.lower()

        # Return True if any keyword is found.
        # The search stops as soon as a match is detected.
        # If no keywords match, search is skipped.
        for word in keywords:
            if word in text_lower:
                return True

        return False