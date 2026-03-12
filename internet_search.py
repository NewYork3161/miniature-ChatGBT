"""
internet_search.py
---------------------------------

This module provides basic internet search functionality
for the MiniChatGPT system.

Its purpose is to retrieve real-world information when the
AI model itself may not know the answer.

For example:

User: "What is the latest news about AI?"

The AI model alone may not know current events, so this
module queries an external search API and returns useful
context that can help generate a better response.

This file uses the DuckDuckGo Instant Answer API because:

• It is free
• It requires no API key
• It returns structured JSON responses
"""

# ---------------------------------------------------------
# Import Required Libraries
# ---------------------------------------------------------

# requests allows the program to make HTTP requests
# to external websites or APIs.
import requests

# Config contains system settings such as
# enabling/disabling search and limiting results.
from config import Config


# ---------------------------------------------------------
# InternetSearch Class
# ---------------------------------------------------------

class InternetSearch:
    """
    This class handles all internet search operations.

    Responsibilities:
    - Query the DuckDuckGo API
    - Extract useful information from results
    - Return summarized context for the AI system
    """

    def __init__(self):
        """
        Constructor runs when the search system is initialized.

        It loads configuration values from config.py.
        """

        # Whether internet search is enabled in the system.
        # This allows the feature to be disabled easily
        # without changing the rest of the code.
        self.enabled = Config.ENABLE_INTERNET_SEARCH

        # Maximum number of search results to return.
        # Prevents extremely long responses.
        self.result_limit = Config.SEARCH_RESULT_LIMIT


    # ---------------------------------------------------------
    # Main Search Function
    # ---------------------------------------------------------

    def search(self, query):
        """
        Perform an internet search and return summarized results.

        Parameters:
        query → the user's question or search phrase

        Returns:
        A string containing summarized search information.

        Example:

        Input:
        "What is artificial intelligence?"

        Output:
        "Artificial intelligence is the simulation of human intelligence..."
        """

        # If search is disabled in config, return nothing.
        if not self.enabled:
            return ""

        try:

            # -------------------------------------------------
            # DuckDuckGo API Endpoint
            # -------------------------------------------------

            url = "https://api.duckduckgo.com/"


            # -------------------------------------------------
            # Search Parameters
            # -------------------------------------------------

            params = {
                # Search query
                "q": query,

                # Request JSON format response
                "format": "json",

                # Prevent redirect results
                "no_redirect": 1,

                # Remove HTML formatting
                "no_html": 1
            }


            # -------------------------------------------------
            # Send HTTP Request
            # -------------------------------------------------

            # requests.get sends the query to the API.
            # timeout prevents the program from hanging
            # if the server takes too long.
            response = requests.get(url, params=params, timeout=5)

            # Convert the returned JSON into a Python dictionary.
            data = response.json()


            # -------------------------------------------------
            # Extract Useful Results
            # -------------------------------------------------

            results = []


            # DuckDuckGo often provides a short summary called
            # "AbstractText". This is typically the best result.
            if data.get("AbstractText"):
                results.append(data["AbstractText"])


            # -------------------------------------------------
            # Extract Related Topics
            # -------------------------------------------------

            # The API may return additional related search results.
            # These are often short informational snippets.

            for topic in data.get("RelatedTopics", []):

                # Some topics contain a "Text" field with useful information.
                if "Text" in topic:
                    results.append(topic["Text"])

                # Stop collecting results if we reached our limit.
                if len(results) >= self.result_limit:
                    break


            # Combine results into a single string.
            return " ".join(results)


        # ---------------------------------------------------------
        # Error Handling
        # ---------------------------------------------------------

        except Exception as e:

            # If debugging is enabled in config.py,
            # print the error message.
            if Config.DEBUG_MODE:
                print(f"[InternetSearch Error] {e}")

            # Return empty context if search fails.
            return ""