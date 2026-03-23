"""
internet_search.py

Provides basic internet search functionality for the chatbot.

Uses the DuckDuckGo Instant Answer API to retrieve simple,
real-world information when needed.
"""

# requests is used to send HTTP requests to external APIs.
# Config provides system settings such as enabling search.
import requests
from config import Config


class InternetSearch:
    """
    Handles internet search requests.

    This class queries an external API and returns summarized
    results. It is used when the system detects that external
    information may improve the response.
    """

    def __init__(self):
        """
        Initialize search settings from configuration.

        Controls whether search is enabled and how many results
        should be returned.
        """

        # Enable or disable search feature.
        # This allows the system to run without external calls.
        # Useful for testing or offline use.
        self.enabled = Config.ENABLE_INTERNET_SEARCH

        # Limit number of results returned.
        # Prevents excessive data from being passed to the model.
        # Keeps responses concise and manageable.
        self.result_limit = Config.SEARCH_RESULT_LIMIT

    def search(self, query):
        """
        Perform a search query and return summarized results.

        Sends a request to DuckDuckGo and extracts useful text.
        Returns a combined string of relevant information.
        """

        # Skip search if feature is disabled.
        # This avoids unnecessary API calls.
        # Returns empty context in that case.
        if not self.enabled:
            return ""

        try:
            # API endpoint for DuckDuckGo Instant Answer.
            # This returns structured JSON data.
            # No API key is required.
            url = "https://api.duckduckgo.com/"

            # Query parameters for the request.
            # Ensures JSON output and cleaner responses.
            # Disables redirects and HTML formatting.
            params = {
                "q": query,
                "format": "json",
                "no_redirect": 1,
                "no_html": 1
            }

            # Send request to API.
            # timeout prevents long waits if the request fails.
            # Response is converted into a Python dictionary.
            response = requests.get(url, params=params, timeout=5)
            data = response.json()

            # Store extracted results.
            # These will be combined into a final context string.
            # Starts empty and fills with useful text.
            results = []

            # Add main abstract if available.
            # This is usually the most relevant summary.
            # Often provides a direct answer.
            if data.get("AbstractText"):
                results.append(data["AbstractText"])

            # Extract related topic snippets.
            # These provide additional supporting information.
            # Stops when result limit is reached.
            for topic in data.get("RelatedTopics", []):
                if "Text" in topic:
                    results.append(topic["Text"])

                if len(results) >= self.result_limit:
                    break

            # Combine all results into one string.
            # This is passed to the inference engine as context.
            # Keeps output simple and usable.
            return " ".join(results)

        except Exception as e:
            # Print error only if debug mode is enabled.
            # Helps diagnose issues during development.
            # Avoids cluttering output in production.
            if Config.DEBUG_MODE:
                print(f"[InternetSearch Error] {e}")

            # Return empty result if anything fails.
            # Ensures the system continues running safely.
            # Prevents crashes from API issues.
            return ""