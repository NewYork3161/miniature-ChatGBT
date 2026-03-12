"""
ai_chat_engine.py
---------------------------------

Core AI controller for MiniChatGPT.

This file acts as the CENTRAL BRAIN of the chatbot system.

It connects multiple subsystems together:

1. Memory System
   Stores conversation history so the AI remembers previous messages.

2. Inference Engine
   Runs the trained neural network model to generate responses.

3. Internet Search
   Optionally retrieves real-world information from the internet
   to improve answers.

Think of this file as the "traffic controller" of the entire AI system.
It receives user input and decides how the system should process it.
"""

# ---------------------------------------------------------
# Import the modules that this engine will coordinate
# ---------------------------------------------------------

# Memory system that stores the conversation history
from memory import Memory

# AI model runtime that generates responses
from inference import InferenceEngine

# Optional internet search system
from internet_search import InternetSearch


# ---------------------------------------------------------
# AIChatEngine Class
# ---------------------------------------------------------

class AIChatEngine:
    """
    Main controller class for the chatbot.

    Responsibilities:
    - Store conversation history
    - Decide when internet search should be used
    - Send user input to the AI model
    - Return generated responses
    """

    def __init__(self):
        """
        Constructor for the AIChatEngine.

        This runs when the chatbot system first starts.
        It initializes the subsystems required for the AI to function.
        """

        # -------------------------------------------------
        # Memory System
        # -------------------------------------------------

        # Memory stores the conversation between the user and the AI.
        # Without this, the AI would forget everything every message.
        #
        # Example memory contents:
        #
        # User: Hello
        # AI: Hi there
        # User: What is AI?
        #
        # This history allows the model to generate contextual replies.
        self.memory = Memory()

        # -------------------------------------------------
        # Inference Engine
        # -------------------------------------------------

        # The inference engine loads the trained neural network
        # and generates predictions based on user input.
        #
        # It performs operations like:
        #
        # text -> tokens -> model -> predicted token -> text
        #
        # This is where the AI "thinks".
        self.inference = InferenceEngine()

        # -------------------------------------------------
        # Internet Search Module
        # -------------------------------------------------

        # This module allows the chatbot to fetch information
        # from the internet when necessary.
        #
        # Example:
        #
        # User: What is the latest news about AI?
        #
        # The engine can query an API and retrieve real-world information.
        #
        # This improves answers when the model lacks knowledge.
        self.search = InternetSearch()


    # ---------------------------------------------------------
    # Main Response Generation Method
    # ---------------------------------------------------------

    def generate_response(self, user_input):
        """
        This method is called whenever the user sends a message.

        Example flow:

        User types message
             ↓
        AIChatEngine receives it
             ↓
        Message stored in memory
             ↓
        Optional internet search performed
             ↓
        Model generates response
             ↓
        Response stored in memory
             ↓
        Response returned to user
        """

        # -------------------------------------------------
        # Step 1: Save User Message to Memory
        # -------------------------------------------------

        # Store the user input so the conversation history
        # grows over time.
        #
        # Example stored entry:
        #
        # { "role": "user", "content": "hello" }
        #
        self.memory.add_user_message(user_input)


        # -------------------------------------------------
        # Step 2: Decide if Internet Search is Needed
        # -------------------------------------------------

        # Default context is empty.
        # Context will contain additional information
        # retrieved from the internet if needed.
        context = ""

        # Check if the message contains keywords that suggest
        # the user is asking about current or factual information.
        #
        # Example:
        #
        # "latest news"
        # "what is quantum computing"
        #
        if self._needs_internet_search(user_input):
            context = self.search.search(user_input)


        # -------------------------------------------------
        # Step 3: Retrieve Conversation History
        # -------------------------------------------------

        # This allows the model to see the previous messages
        # in the conversation and respond more intelligently.
        #
        # Example:
        #
        # User: What is AI?
        # AI: Artificial intelligence is...
        #
        # User: What about machine learning?
        #
        # The AI can understand the context of the conversation.
        history = self.memory.get_history()


        # -------------------------------------------------
        # Step 4: Generate AI Response
        # -------------------------------------------------

        # The inference engine is responsible for actually
        # generating the response using the trained neural network.
        #
        # Inputs provided to the model include:
        #
        # - current user message
        # - conversation history
        # - optional internet search results
        #
        response = self.inference.generate(user_input, history, context)


        # -------------------------------------------------
        # Step 5: Save AI Response to Memory
        # -------------------------------------------------

        # Store the AI response so future messages
        # include this in conversation history.
        #
        # Example stored entry:
        #
        # { "role": "ai", "content": "Hello! How can I help?" }
        #
        self.memory.add_ai_message(response)


        # -------------------------------------------------
        # Step 6: Return Response
        # -------------------------------------------------

        # The response is returned to the main program
        # (main.py or interface.py) where it is displayed
        # to the user.
        return response


    # ---------------------------------------------------------
    # Internet Search Decision Logic
    # ---------------------------------------------------------

    def _needs_internet_search(self, text):
        """
        Determines whether the system should perform
        an internet search.

        Currently this is a simple rule-based trigger
        based on keywords.

        In a more advanced AI system this would be replaced by:
        - intent classification
        - semantic analysis
        - or another neural model
        """

        # Keywords that suggest the user wants factual
        # or current information.
        keywords = [
            "latest",
            "news",
            "today",
            "current",
            "recent",
            "who is",
            "what is"
        ]

        # Convert message to lowercase so comparisons
        # are case-insensitive.
        text_lower = text.lower()

        # Check if any keyword appears in the message.
        for word in keywords:
            if word in text_lower:
                return True

        # If no keywords are found, internet search
        # is not required.
        return False