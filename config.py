class Config:
    """
    Central configuration class for the chatbot system.

    This class stores all constants related to model training, dataset paths,
    feature flags, and runtime limits. Keeping everything in one place makes
    it easier to manage and adjust behavior without changing multiple files.
    """

    # ---------------------------------
    # Model + Training Settings
    # ---------------------------------

    # Path where the trained model is saved and loaded from.
    # This allows the system to reuse a trained model instead of retraining.
    # The file is expected to be created after the training process completes.
    MODEL_PATH = "models/minichat_model.pt"

    # Size of the embedding vectors used to represent tokens.
    # Larger values can capture more meaning but increase computation cost.
    # This directly affects how text is represented inside the model.
    EMBED_SIZE = 256

    # Size of the hidden layer in the neural network.
    # This controls the model's capacity to learn patterns.
    # Higher values increase power but also memory and compute usage.
    HIDDEN_SIZE = 512

    # Number of tokens per training sequence.
    # This defines how much context the model sees at once.
    # Longer sequences improve context but require more resources.
    SEQ_LENGTH = 100

    # Number of samples processed in each training batch.
    # Larger batches improve training stability but use more memory.
    # This value should be adjusted based on available hardware.
    BATCH_SIZE = 64

    # Total number of passes through the dataset during training.
    # More epochs allow the model to learn better but risk overfitting.
    # This value controls how long training runs.
    EPOCHS = 50

    # Learning rate used by the optimizer.
    # This controls how quickly the model updates its weights.
    # If too high, training becomes unstable; if too low, training is slow.
    LEARNING_RATE = 0.0005


    # ---------------------------------
    # Dataset
    # ---------------------------------

    # Path to the dataset used for training the model.
    # This file should contain the raw text data to be processed.
    # The tokenizer and dataset loader will read from this location.
    DATASET_PATH = "data/shakespeare.txt"


    # ---------------------------------
    # Feature Toggles
    # ---------------------------------

    # Enables additional logging and debugging behavior.
    # Useful during development to trace issues and inspect values.
    # Should typically be disabled in production environments.
    DEBUG_MODE = True

    # Enables or disables the internet search feature.
    # When disabled, the model relies only on its training and memory.
    # This can improve performance and avoid external dependencies.
    ENABLE_INTERNET_SEARCH = False


    # ---------------------------------
    # Limits / Memory
    # ---------------------------------

    # Maximum number of results returned from the search module.
    # This prevents excessive data from being passed into the model.
    # Helps keep responses focused and efficient.
    SEARCH_RESULT_LIMIT = 5

    # Maximum number of messages stored in conversation memory.
    # Older messages are removed once this limit is exceeded.
    # This controls memory usage and keeps context manageable.
    MAX_MEMORY_MESSAGES = 10