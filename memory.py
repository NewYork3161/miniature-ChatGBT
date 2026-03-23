"""
memory.py

Handles conversation memory for the chatbot.

Stores recent user and AI messages so the system
can maintain context across multiple interactions.
"""

from config import Config


class Memory:
    """
    Manages conversation history for the chatbot.

    Stores messages in order and ensures the total
    number of stored messages stays within a limit.
    """

    def __init__(self):
        """
        Initialize memory storage and limits.

        Sets up an empty history list and loads
        the maximum message limit from configuration.
        """

        # List to store conversation history.
        # Each entry contains a role and message content.
        # Order is preserved for context.
        self.history = []

        # Maximum number of messages to keep.
        # Older messages are removed when limit is exceeded.
        # Helps control memory usage and context size.
        self.max_messages = Config.MAX_MEMORY_MESSAGES

    def add_user_message(self, message):
        """
        Add a user message to memory.

        Stores the message with role "user" and
        ensures memory stays within size limits.
        """

        # Create structured entry for user message.
        # This keeps format consistent across all messages.
        # Required for downstream processing.
        entry = {
            "role": "user",
            "content": message
        }

        # Append message to history.
        # Then trim memory if needed.
        # Keeps most recent messages only.
        self.history.append(entry)
        self._trim_memory()

    def add_ai_message(self, message):
        """
        Add an AI response to memory.

        Stores the message with role "ai" and
        maintains the memory size constraint.
        """

        # Create structured entry for AI response.
        # Matches format used for user messages.
        # Ensures consistency in history.
        entry = {
            "role": "ai",
            "content": message
        }

        # Append message to history.
        # Then trim memory if needed.
        # Ensures memory does not grow indefinitely.
        self.history.append(entry)
        self._trim_memory()

    def get_history(self):
        """
        Return the current conversation history.

        This is used by the inference engine to
        provide context for generating responses.
        """

        # Return full history list.
        # Contains both user and AI messages.
        # Order reflects conversation flow.
        return self.history

    def clear(self):
        """
        Clear all stored conversation history.

        Resets memory to an empty state.
        Useful for starting a new session.
        """

        # Reset history list.
        # Removes all stored messages.
        # Frees memory for new conversation.
        self.history = []

    def _trim_memory(self):
        """
        Ensure memory does not exceed maximum size.

        Removes oldest messages when limit is reached.
        Keeps only the most recent entries.
        """

        # Check if memory exceeds limit.
        # If so, slice to keep only recent messages.
        # This maintains a fixed-size history.
        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]