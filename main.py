"""
main.py

Entry point for the MiniChatGPT application.

Handles user interaction, initializes the AI engine,
and runs the main chat loop.
"""

# Import the core AI engine.
# This is the only component main.py interacts with directly.
# All internal logic is handled inside AIChatEngine.
from ai_chat_engine import AIChatEngine


def start_chat():
    """
    Start the console-based chat interface.

    Initializes the AI engine and continuously processes
    user input until the program is exited.
    """

    # Display header for the chat session.
    # Provides basic instructions to the user.
    # Keeps the interface simple and readable.
    print("\n===================================")
    print("        MiniChatGPT Console")
    print("===================================")
    print("Type 'exit' or 'quit' to stop the program.\n")

    # Initialize the AI engine.
    # This sets up memory, inference, and optional search.
    # The engine will handle all message processing.
    engine = AIChatEngine()

    # Main loop for continuous interaction.
    # Runs until the user exits or an interruption occurs.
    # Each iteration processes one user message.
    while True:
        try:
            # Get user input from console.
            # strip() removes extra spaces for cleaner handling.
            # Input is expected as plain text.
            user_input = input("You: ").strip()

            # Check for exit commands.
            # Allows user to stop the program cleanly.
            # Comparison is case-insensitive.
            if user_input.lower() in ["exit", "quit"]:
                print("\nMiniChatGPT shutting down.")
                break

            # Skip empty input.
            # Prevents unnecessary processing.
            # Keeps loop efficient.
            if user_input == "":
                continue

            # Send input to AI engine.
            # The engine handles memory, search, and inference.
            # Returns the generated response.
            response = engine.generate_response(user_input)

            # Display AI response.
            # Output is printed to console.
            # Formatting keeps responses readable.
            print(f"AI: {response}\n")

        except KeyboardInterrupt:
            # Handle Ctrl+C interruption.
            # Allows graceful shutdown.
            # Prevents abrupt termination.
            print("\n\nProgram interrupted. Exiting.")
            break

        except Exception as e:
            # Handle unexpected errors.
            # Displays error message without crashing.
            # Useful during development and debugging.
            print(f"\nError: {e}\n")


def main():
    """
    Entry function for the application.

    Starts the chat interface when the program runs.
    """
    start_chat()


# Run main only if this file is executed directly.
# Prevents the chat loop from starting on import.
# Standard Python entry point pattern.
if __name__ == "__main__":
    main()