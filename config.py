class Config:

    # ---------------------------------
    # Model + Training Settings
    # ---------------------------------

    MODEL_PATH = "models/minichat_model.pt"

    EMBED_SIZE = 256
    HIDDEN_SIZE = 512

    SEQ_LENGTH = 100

    BATCH_SIZE = 64

    EPOCHS = 50

    LEARNING_RATE = 0.0005


    # ---------------------------------
    # Dataset
    # ---------------------------------

    DATASET_PATH = "data/shakespeare.txt"


    # ---------------------------------
    # Feature Toggles
    # ---------------------------------

    DEBUG_MODE = True
    ENABLE_INTERNET_SEARCH = False


    # ---------------------------------
    # Limits / Memory
    # ---------------------------------

    SEARCH_RESULT_LIMIT = 5
    MAX_MEMORY_MESSAGES = 10