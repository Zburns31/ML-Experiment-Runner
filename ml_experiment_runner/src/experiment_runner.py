# Import logging first so other libraries don't take precedence for logging
import logging
import os
from config import Config
from pathlib import Path

# Used for loggin purposes
width = os.get_terminal_size().columns

config = Config(verbose=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    handlers=[
        logging.FileHandler(Path(config.LOGS_DIR, "ml_experiments.log"), mode="w+"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import json

from sklearn.base import BaseEstimator
from typing import Self, List, Dict, Type, Union
from datetime import datetime
from collections import defaultdict

from learners.base_learner import BaseClassifier
from learners.DT import DTClassifier
from config import Config
from dataset import Dataset


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

        return self.model.name.replace(" ", "_") + "_" + self.data.dataset_name

    def run_experiment(
        self,
        X: np.array,
        y: np.array,
        param_name: str,
        param_range: float,
    ) -> None:
        experiment_times = {}

        if self.config.VERBOSE:
            logger.info(
                f"Running Experiment: {self.model.name} | Parameter Name: {param_name} = {param_range.tolist()}"
            )

        start = datetime.now()

        estimator = self.model_class(self.config, self.param_grid, self.eval_metric)
        if self.config.VERBOSE:
            logger.info(json.dumps(estimator.get_params(), indent=4))

        best_param_value = estimator.plot_validation_curve(
            X,
            y,
            dataset_name=self.data.dataset_name,
            param_name=param_name,
            param_range=param_range,
            save_plot=True,
        )

        estimator.set_params(**{param_name: best_param_value})
        estimator.plot_learning_curve(
            X,
            y,
            param_name=param_name,
            dataset_name=self.data.dataset_name,
            save_plot=True,
        )
        end = datetime.now()
        run_time = end - start

        experiment_times[f"{estimator.name}_{param_name}"] = run_time.seconds
        return experiment_times

    def main(self) -> Dict[str, float]:
        self.model = self.init_estimator()

        logger.info(
            f"Estimator Hyperparameter Grid: \n{json.dumps(self.param_grid, default = str, indent = 4)}"
        )

        logger.info(
            f"Starting Experiments for: {self.model.name} | Dataset Name: {self.data.dataset_name}"
        )

        logger.info(f"Experiment Name: {self.experiment_name}")

        experiment_details = defaultdict(list)
        for param_name, param_range in self.param_grid.items():

            experiment_details[self.model.name].append(
                self.run_experiment(
                    self.data.features, self.data.target, param_name, param_range
                )
            )
            logger.info("-" * width)

        return experiment_details


if __name__ == "__main__":
    wine_data = Dataset("winequality-white.csv", data_delimiter=";", config=config).run(
        target_col="quality"
    )

    logger.info("Finished Processing Dataset")

    logger.info("Beginning ML Experiments")

    eval_metric = "accuracy"
    # Outlines what experiments we want to run. These get passed to the underlying estimator
    param_grid = {
        "max_depth": np.arange(1, 21),
        "ccp_alpha": np.arange(0.1, 1, 0.1),
        "min_samples_leaf": np.arange(1, 101, 10),
    }

    dt_experiment = MLExperimentRunner(
        DTClassifier, wine_data, config, eval_metric, param_grid
    )
    logger.info(f"Experiment Name: {dt_experiment.experiment_name}")

    experiment_times_dict = dt_experiment.main()
    logger.info(json.dumps(dict(experiment_times_dict), indent=4))
