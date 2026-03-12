"""
main.py
---------------------------------

This file is the entry point for the MiniChatGPT system.

When the program is started, THIS file runs first.

Its responsibilities are:

1. Start the chat interface
2. Create the AI engine
3. Accept user input
4. Send user input to the AI system
5. Display AI responses
6. Handle exit commands and errors

Think of this file as the "front door" of the AI application.
"""


# ---------------------------------------------------------
# Import the Core AI Engine
# ---------------------------------------------------------

# AIChatEngine is the central controller that connects:
# - memory
# - inference (AI model)
# - internet search
#
# main.py communicates ONLY with this class.
# Everything else happens behind the scenes.
from ai_chat_engine import AIChatEngine


# ---------------------------------------------------------
# Chat Interface Function
# ---------------------------------------------------------

def start_chat():
    """
    This function starts the console chat interface.

    It displays the chat header, creates the AI engine,
    and then continuously waits for user input.

    The chat runs in a loop until the user exits.
    """

    # Print chat header
    print("\n===================================")
    print("        MiniChatGPT Console")
    print("===================================")
    print("Type 'exit' or 'quit' to stop the program.\n")

    # -----------------------------------------------------
    # Initialize AI Engine
    # -----------------------------------------------------

    # Create the AI system that will process messages.
    #
    # Internally this initializes:
    #
    # AIChatEngine
    #     ↓
    # Memory system
    # Inference engine (neural network)
    # Internet search module
    #
    engine = AIChatEngine()

    # -----------------------------------------------------
    # Main Chat Loop
    # -----------------------------------------------------

    # This loop keeps the chatbot running.
    while True:

        try:

            # -------------------------------------------------
            # Get User Input
            # -------------------------------------------------

            # Prompt the user to type a message.
            # .strip() removes leading/trailing spaces.
            user_input = input("You: ").strip()


            # -------------------------------------------------
            # Exit Conditions
            # -------------------------------------------------

            # If the user types exit or quit,
            # stop the program.
            if user_input.lower() in ["exit", "quit"]:
                print("\nMiniChatGPT shutting down.")
                break


            # If the user entered nothing, skip the loop
            if user_input == "":
                continue


            # -------------------------------------------------
            # Send Message to AI Engine
            # -------------------------------------------------

            # The message is sent to AIChatEngine,
            # which processes it through several steps:
            #
            # 1. Save message in memory
            # 2. Possibly perform internet search
            # 3. Run neural network inference
            # 4. Generate AI response
            #
            response = engine.generate_response(user_input)


            # -------------------------------------------------
            # Display AI Response
            # -------------------------------------------------

            print(f"AI: {response}\n")


        # -----------------------------------------------------
        # Handle Ctrl+C interruption
        # -----------------------------------------------------

        except KeyboardInterrupt:

            # If the user presses Ctrl+C,
            # gracefully exit the program.
            print("\n\nProgram interrupted. Exiting.")
            break


        # -----------------------------------------------------
        # General Error Handling
        # -----------------------------------------------------

        except Exception as e:

            # If something unexpected happens,
            # show the error instead of crashing silently.
            print(f"\nError: {e}\n")


# ---------------------------------------------------------
# Main Program Entry Point
# ---------------------------------------------------------

def main():
    """
    Main program launcher.

    This simply calls start_chat().
    """
    start_chat()


# ---------------------------------------------------------
# Python Execution Guard
# ---------------------------------------------------------

# This ensures that main() runs ONLY when the script
# is executed directly.

# If this file is imported by another Python file,
# the main chat loop will NOT start automatically.

if __name__ == "__main__":
    main()