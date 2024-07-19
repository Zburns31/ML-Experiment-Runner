RANDOM_SEED = 42
IMAGE_DIR = "images/"
RESULTS_DIR = "results/"
DATA_DIR = "data/"
LOGS_DIR = "logs/"


class Config:
    RANDOM_SEED = 42
    IMAGE_DIR = "images/"
    RESULTS_DIR = "results/"
    DATA_DIR = "data/"
    LOGS_DIR = "logs/"

    ML_PREPROCESS_PARAMS = {
        "shuffle": True,
        "stratify": True,
        "test_size": 0.3,
        "class_weights": None,
    }

    def __init__(self, verbose: bool = False):
        self.VERBOSE = verbose
