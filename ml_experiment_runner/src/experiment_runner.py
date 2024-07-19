import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from typing import Self, List, Dict, Type, Union
from pathlib import Path
import logging
from datetime import datetime

from learners.base_learner import BaseClassifier
from learners.DT import DTClassifier
from config import Config
from dataset import Dataset

config = Config()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    handlers=[
        logging.FileHandler(Path(config.LOGS_DIR, "ml_experiments.log"), mode="w+"),
        logging.StreamHandler(),
    ],
)


class MLExperimentRunner:
    def __init__(
        self,
        model_class: Type[Union[BaseClassifier, BaseEstimator]],
        data: Dataset,
        config: Type[Config],
        eval_metric: str,
        param_grid: Dict[str, float],
        test_size: float = 0.3,
    ):
        self.model_class = model_class
        self.model = None
        self.data = data
        self.config = config
        self.param_grid = param_grid
        self.eval_metric = eval_metric
        self.seed = config.RANDOM_SEED
        self.verbose = config.VERBOSE
        self.test_size = test_size

    def init_estimator(self) -> Union[BaseClassifier, BaseEstimator]:
        return self.model_class(self.config, self.eval_metric, self.param_grid)

    @property
    def experiment_name(self) -> str:
        # Check if underlying model has been instantiated
        if not self.model:
            self.model = self.init_estimator()

        return self.model.name.replace(" ", "_") + "_" + self.data.name

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

    def run_experiment(
        self,
        X: np.array,
        y: np.array,
        param_name: str,
        param_range: float,
        verbose=False,
    ) -> None:
        experiment_times = {}

        if verbose:
            logger.info(
                f"Running Experiment: {self.model.name} | Parameter Name: {param_name} = {list(param_range)}"
            )

        start = datetime.now()

        estimator = self.model_class(self.config, self.param_grid, self.eval_metric)
        if verbose:
            logger.info(estimator.get_params())

        scoring_func = estimator.get_scorer(self.eval_metric)

        best_param_value = estimator.plot_validation_curve(
            X,
            y,
            dataset_name=self.data.name,
            param_name=param_name,
            param_range=param_range,
            save_plot=True,
        )

        estimator.set_params(max_depth=best_param_value)
        estimator.plot_learning_curve(
            X, y, param_name=param_name, dataset_name=self.data.name, save_plot=True
        )
        end = datetime.now()
        run_time = end - start

        experiment_times[f"{estimator.name}_{param_name}"] = run_time.seconds
        return experiment_times

    def main(self) -> Dict[str, float]:
        self.model = self.init_estimator()

        logger.info(
            f"Starting Experiments for: {self.model.name} | Dataset Name: {self.data.name}"
        )
        experiment_details = {}
        for param_name, param_range in self.param_grid.items():
            experiment_details[self.model.name] = self.run_experiment(
                X, y, param_name, param_range, self.verbose
            )

        return experiment_details


if __name__ == "__main__":
    wine_data = Dataset("winequality-white.csv", config)
    wine_data.load_data(delimiter=";")
    wine_data.summary_statistics("quality")
    wine_data.check_missing_values()
    outliers_dict = wine_data.check_outliers()

    X_TRAIN, X_TEST, y_train, y_test, X, y = wine_data.create_train_test_split(
        "quality", 0.3, 42, stratify_col="quality"
    )

    logger.info("Finished Processing Dataset")

    eval_metric = "accuracy"
    # Outlines what experiments we want to run. These get passed to the underlying estimator
    param_grid = {
        "max_depth": np.arange(1, 21),
        "ccp_alpha": np.arange(0.1, 1, 0.1),
        "min_samples_per_leaf": np.arange(1, 101, 10),
    }
    print(param_grid)
    dt_experiment = MLExperimentRunner(
        DTClassifier, wine_data, config, eval_metric, param_grid
    )
    logger.info(f"Starting Experiments for: {dt_experiment.experiment_name}")

    dt_experiment.run_experiment(
        X.values, y.values, "max_depth", np.arange(1, 21), verbose=True
    )
