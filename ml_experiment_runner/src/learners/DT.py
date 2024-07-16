import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import logging
import time

from config import Config
from utilities import get_directory
from .base_learner import BaseClassifier
from typing import Dict, Tuple, List, Self, Type
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    validation_curve,
    cross_val_score,
)


class DTClassifier(BaseClassifier):
    def __init__(
        self,
        config: Type[Config],
        param_grid: Dict[str, List[int]],
        eval_metric: str,
    ):
        super().__init__(
            model=DecisionTreeClassifier(random_state=config.RANDOM_SEED),
            config=config,
            param_grid=param_grid,
            eval_metric=eval_metric,
            seed=config.RANDOM_SEED,
            verbose=config.VERBOSE,
        )

    def get_model_name(self) -> str:
        return "Decision Tree Classifier"

    def fit(self, X, y, verbose=True):
        if verbose:
            print("Fitting the model...")
        self.model.fit(X, y)
        if verbose:
            print("Model fitting completed.")
        return self

    def predict(self, X):
        return self.model.predict(X)

    def plot_learning_curve(
        self,
        X: np.array,
        y: np.array,
        param_name: str,
        dataset_name: str,
        cv: int = 5,
        save_plot=True,
        show_plot=False,
    ):
        train_sizes, train_scores, test_scores = learning_curve(self.model, X, y, cv=cv)

        train_scores_mean = np.mean(train_scores, axis=1) * 100
        test_scores_mean = np.mean(test_scores, axis=1) * 100

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), dpi=100, sharey=False)
        annotation = (
            param_name.replace("_", " ").capitalize()
            + " = "
            + str(self.model.get_params().get(param_name))
        )
        plt.title(f"Learning Curve ({self.get_model_name()}) | {annotation}")
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
            model_name = self.get_model_name().replace(" ", "_")
            plot_name = f"{dataset_name}_{model_name}_learning_curve.png"

            image_path = Path(get_directory(self.config.IMAGE_DIR), plot_name)
            plt.savefig(image_path)
            print(image_path)

        if show_plot:
            plt.show()

    def plot_validation_curve(
        self,
        X,
        y,
        dataset_name,
        param_name,
        param_range,
        cv=5,
        save_plot=True,
        show_plot=False,
    ) -> int:
        """
        Plot a validation curve with the range of hyperparameter values on the X-axis and the metric score on the Y-axis.

        :param estimator: The model/estimator for which the validation curve is plotted.
        :param X: The input data.
        :param y: The target data.
        :param param_name: Name of the hyperparameter to vary.
        :param param_range: The range of values for the hyperparameter.
        :param scoring: The scoring function to use.
        :param cv: The number of cross-validation folds (default is 5).
        :param n_jobs: The number of jobs to run in parallel (default is -1, using all processors).
        :param verbose: The verbosity level (default is 1).

        returns the best hyperparameter value
        """
        # scoring=scoring
        train_scores, test_scores = validation_curve(
            self.model,
            X,
            y,
            param_name=param_name,
            param_range=param_range,
            cv=cv,
            verbose=self.verbose,
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
            model_name = self.get_model_name().replace(" ", "_")
            plot_name = f"{dataset_name}_{model_name}_validation_curve.png"

            image_path = Path(get_directory(self.config.IMAGE_DIR), plot_name)
            plt.savefig(image_path)
            print(image_path)

        if show_plot:
            plt.show()

        return int(best_param_value)

    def plot_training_run_time(self, X, y, cv=5, show_plot=False):
        start_time = time.time()
        scores = cross_val_score(self.model, X, y, cv=cv)
        end_time = time.time()

        runtime = end_time - start_time

        plt.figure()
        plt.title(f"Training Runtime ({self.get_human_readable_model_name()})")
        plt.xlabel("Training examples")
        plt.ylabel("Runtime (seconds)")
        plt.bar(["Run Time"], [runtime])

        if show_plot:
            plt.show()
