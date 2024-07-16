from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from typing import Self, List, Dict, Type, Union
import logging

from learners.base_learner import BaseClassifier
from config import Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLExperimentRunner(BaseClassifier):
    def __init__(
        self,
        model: BaseClassifier,
        config: Type[Config],
        eval_metric: str,
        param_grid: Dict[str, float],
        search_method: str,
        experiment_name: str,
    ):
        self.model = model
        self.config = config
        self.param_grid = param_grid
        self.eval_metric = eval_metric
        self.search_method = search_method
        self.experiment_name = experiment_name
        # Set to model_name_log.py
        self.logging_filename = None
        self.seed = config.RANDOM_SEED
        self.verbose = config.verbose

    def save_training_results(self, learner: BaseEstimator) -> None:
        pass

    def save_test_results(self, learner: BaseEstimator) -> None:
        pass

    def log(self, msg, *args):
        """
        If the experiment is set to Verbose = True, log the message and the passed in arguments
        """
        if self._verbose:
            logger.info(msg.format(*args))

    def run_experiment(self, *args) -> None:
        pass

    def main(self) -> None:
        pass
