"""
memory.py
---------------------------------
Handles conversation memory for MiniChatGPT.

Stores recent user and AI messages so the system
can maintain conversational context.
"""

from config import Config


class Memory:

    def __init__(self):

        # Stores conversation history
        self.history = []

        # Maximum messages allowed in memory
        self.max_messages = Config.MAX_MEMORY_MESSAGES


    def add_user_message(self, message):
        """
        Add a user message to memory
        """

        entry = {
            "role": "user",
            "content": message
        }

        self.history.append(entry)
        self._trim_memory()


    def add_ai_message(self, message):
        """
        Add an AI response to memory
        """

        entry = {
            "role": "ai",
            "content": message
        }

        self.history.append(entry)
        self._trim_memory()


    def get_history(self):
        """
        Return the current conversation history
        """

        return self.history


    def clear(self):
        """
        Clear conversation memory
        """

        self.history = []


    def _trim_memory(self):
        """
        Ensures memory does not exceed the max size
        """

        if len(self.history) > self.max_messages:
            self.history = self.history[-self.max_messages:]