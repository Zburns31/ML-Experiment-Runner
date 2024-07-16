from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from typing import Self, List, Dict
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLExperimentRunner(ABC, BaseEstimator):
    def __init__(
        self,
        model,
        config,
        eval_metric,
        param_grid,
        search_method,
        experiment_name,
        verbose=False,
    ):
        self.model = model
        self.config = config
        self.param_grid = param_grid
        self.eval_metric = eval_metric
        self.search_method = search_method
        self.best_model = None
        self.experiment_name = experiment_name
        # Set to model_name_log.py
        self.logging_filename = None
        self.seed = 42
        self._verbose = verbose

    @abstractmethod
    def get_human_readable_model_name(self) -> str:
        pass

    @abstractmethod
    def fit(self, X, y, verbose=True) -> Self:
        pass

    @abstractmethod
    def predict(self, X, y) -> Self:
        pass

    @abstractmethod
    def plot_learning_curve(self, learner: BaseEstimator) -> None:
        pass

    @abstractmethod
    def plot_learning_run_time(self, learner: BaseEstimator) -> None:
        pass

    @abstractmethod
    def save_training_results(self, learner: BaseEstimator) -> None:
        pass

    @abstractmethod
    def save_test_results(self, learner: BaseEstimator) -> None:
        pass

    @abstractmethod
    def run_experiment(self, *args) -> None:
        pass

    def log(self, msg, *args):
        """
        If the experiment is set to Verbose = True, log the message and the passed in arguments
        """
        if self._verbose:
            logger.info(msg.format(*args))
