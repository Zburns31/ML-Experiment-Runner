import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from config import Config
from pathlib import Path
from utilities import get_directory
from abc import ABC, abstractmethod
from typing import Dict, Type, List, Self, Any, Callable, Union

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
)


class BaseClassifier(ClassifierMixin, BaseEstimator, ABC):
    """
    TODO: Nest directories for images and results for easier sorting

    Args:
        ClassifierMixin (_type_): _description_
        BaseEstimator (_type_): _description_
        ABC (_type_): _description_
    """

    def __init__(
        self,
        model: BaseEstimator,
        config: Type[Config],
        param_grid: Dict[str, List[int]],
        eval_metric: str,
    ):
        self.model = model
        self.config = config
        self.param_grid = param_grid
        self.eval_metric = eval_metric
        self.seed = config.RANDOM_SEED
        self.verbose = config.VERBOSE

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def classes_(self):
        return self._learner.classes_

    @property
    def n_classes_(self):
        return self._learner.n_classes_

    @property
    def unique_hyperparameters(self) -> List[str]:
        return list(self.param_grid.keys())

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get the current parameters for the learner. This passes the call back to the learner from learner()

        :param deep: If true, fetch deeply
        :return: The parameters
        """
        return self.model.get_params(deep)

    def set_params(self, **params) -> None:
        """
        Set the current parameters for the learner. This passes the call back to the learner from learner()

        :param params: The params to set
        :return: self
        """
        print(f"New Parameters Set: {params}")
        self.model.set_params(**params)

    def get_scorer(self, metric_name: str) -> Callable:
        """
        Given a metric name, return the corresponding sklearn scoring function

        :param metric_name: A string representing the metric name (e.g., 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'confusion_matrix')
        :return: A callable scoring function from sklearn.metrics
        """
        scorers = {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "fbeta": fbeta_score,
            "roc_auc": roc_auc_score,
        }

        if metric_name in scorers:
            return make_scorer(scorers[metric_name])
        else:
            raise ValueError(
                f"Unknown metric name: {metric_name}. Valid options are: {list(scorers.keys())}"
            )

    @abstractmethod
    def fit(self, X, y, verbose=True) -> Self:
        pass

    @abstractmethod
    def predict(self, X, y, proba=False) -> Self:
        pass

    def plot_learning_curve(
        self,
        X: np.array,
        y: np.array,
        param_name: str,
        dataset_name: str,
        cv: int = 5,
        save_plot: bool = True,
        show_plot: bool = False,
    ):
        """Generates a learning curve for the underlying model

        TODO: Add parameter for custom training set sizes

        Args:
            X (np.array): Represnets the predictor/independent features
            y (np.array): Represents the target/repsonse variable
            param_name (str): Hyperparameter that we are using in the underlying model
            dataset_name (str): Name of the dataset we are training/predicting against
            cv (int, optional): Number of cross-validation folds. Defaults to 5.
            save_plot (bool, optional): Whether to save the generated charts or not. Defaults to True.
            show_plot (bool, optional): Whether to plot the generated charts or not. Defaults to False.
        """
        train_sizes, train_scores, test_scores = learning_curve(self.model, X, y, cv=cv)

        train_scores_mean = np.mean(train_scores, axis=1) * 100
        test_scores_mean = np.mean(test_scores, axis=1) * 100

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=100, sharey=False)
        annotation = (
            param_name.replace("_", " ").capitalize()
            + " = "
            + str(self.model.get_params().get(param_name))
        )
        plt.title(f"Learning Curve ({self.name}) | {annotation}")
        plt.xlabel("# of Training Observations")
        plt.ylabel("Score")

        plt.plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Cross-validation score",
        )
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.legend(loc="best")
        plt.tight_layout()

        if save_plot:
            model_name = self.name.replace(" ", "_")
            plot_name = f"{dataset_name}_{model_name}_learning_curve.png"

            image_path = Path(
                get_directory(
                    self.config.ARTIFACTS_DIR,
                    self.config.IMAGE_DIR,
                    dataset_name,
                    model_name,
                ),
                # get_directory(self.config.IMAGE_DIR, dataset_name, model_name),
                plot_name,
            )
            plt.savefig(image_path)

            if self.verbose:
                print(f"Saving Learning Curve to: {image_path.relative_to(Path.cwd())}")

        if show_plot:
            plt.show()

    def plot_validation_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str,
        param_name: str,
        param_range: Union[np.ndarray, List[float]],
        cv: int = 5,
        save_plot: bool = True,
        show_plot: bool = False,
    ) -> int:
        """Plot a validation curve with the range of hyperparameter values on the X-axis and the metric score on the Y-axis. This function
        also returns the value of the specified hyperparameter with the best testing score

        Args:
            X (np.array): Represnets the predictor/independent features
            y (np.array): Represents the target/repsonse variable
            dataset_name (str): Name of the dataset we are training/predicting against
            param_name (str): Hyperparameter that we are using in the underlying model
            param_range ()
            cv (int, optional): Number of cross-validation folds. Defaults to 5.
            save_plot (bool, optional): Whether to save the generated charts or not. Defaults to True.
            show_plot (bool, optional): Whether to plot the generated charts or not. Defaults to False.

        Returns:
            int: The value of the specified hyperparameter that returns the best mean test score
        """

        train_scores, test_scores = validation_curve(
            self.model,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
        )

        train_scores_mean = np.mean(train_scores, axis=1) * 100
        train_scores_std = np.std(train_scores, axis=1) * 100
        test_scores_mean = np.mean(test_scores, axis=1) * 100
        test_scores_std = np.std(test_scores, axis=1) * 100

        # Get the best parameter value based on the cross-validation score
        best_param_index = np.argmax(test_scores_mean)
        best_param_value = param_range[best_param_index]

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=100, sharey=False)
        cleaned_param_name = param_name.replace("_", " ").capitalize()
        plt.title(f"Validation Curve for {param_name}")
        plt.xlabel(cleaned_param_name)
        plt.ylabel("Score")

        plt.plot(param_range, train_scores_mean, label="Training score", color="r")
        plt.fill_between(
            param_range,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )

        plt.plot(
            param_range, test_scores_mean, label="Cross-validation score", color="g"
        )
        plt.fill_between(
            param_range,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )

        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.legend(loc="best")
        plt.tight_layout()

        if save_plot:
            model_name = self.name.replace(" ", "_")
            plot_name = f"{dataset_name}_{model_name}_validation_curve.png"

            image_path = Path(
                get_directory(
                    self.config.ARTIFACTS_DIR,
                    self.config.IMAGE_DIR,
                    dataset_name,
                    model_name,
                ),
                plot_name,
            )
            plt.savefig(image_path)

            if self.verbose:
                # Relative path
                print(
                    f"Saving Validation Curve to: {image_path.relative_to(Path.cwd())}"
                )

        if show_plot:
            plt.show()

        return int(best_param_value)

    def plot_training_run_time(self, learner: BaseEstimator) -> None:
        """
        TODO: Move implementation from DT into here

        Args:
            learner (BaseEstimator): _description_
        """
        pass

    def plot_confusion_matrix(self):
        pass

    def plot_roc_curve(self, binary_clf=False):
        pass
