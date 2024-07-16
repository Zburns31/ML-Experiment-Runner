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

    def __init__(self, verbose: bool = False):
        self.VERBOSE = verbose
