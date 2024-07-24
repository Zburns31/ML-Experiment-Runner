import argparse
import os
import json
import numpy as np

from typing import Type, List, Union, Dict
from dataset import Dataset
from experiment_runner import MLExperimentRunner
from learners.DT import DTClassifier
from learners.base_learner import BaseClassifier
from experiment_runner import MLExperimentRunner
from config import Config
from utilities import print_tuples

# Used for logging purposes
width = os.get_terminal_size().columns

###############################################################################
# ML Experiment Parameter Configurations

# Decision Tree
DT_EVAL_METRIC = "accuracy"
# Outlines what experiments we want to run. These get passed to the underlying estimator
DT_PARAM_GRID = {
    "max_depth": np.arange(1, 21),
    "ccp_alpha": np.arange(0.1, 1, 0.1),
    "min_samples_leaf": np.arange(1, 101, 10),
}


###############################################################################
def run_experiment_configuration(
    datasets: List[Dataset],
    estimator: MLExperimentRunner,
    eval_metric: str,
    param_grid: Dict[str, float],
    config: Config,
) -> List[dict]:
    """
    Runs the provided MLExperimentRunner against all datasets passed in

    Args:
        datasets (List[Dataset]): Dataset objects to fit the models against
        estimator (MLExperimentRunner): Learning algorithm to run experiment methods against
        eval_metric (str): Evaluation metric
        param_grid (Dict[str, float]): Hyperparameters and associated ranges to run experiments against
        config (Config): Project configuration settings

    Returns:
        List[dict]: Experiment run times for each configuration
    """

    experiment_times = []

    for dataset in datasets:
        experiment = MLExperimentRunner(
            estimator, dataset, config, eval_metric, param_grid
        )
        if config.VERBOSE:
            print(f"Experiment Name: {experiment.experiment_name}")

        experiment_times_dict = experiment.main()
        experiment_times.append(dict(experiment_times_dict))
        print("-" * width)

    return experiment_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ML Experiment Runner", description="Perform some ML experiments"
    )

    # Define the valid choices for the experiment type
    experiment_choices = ["dt", "ann", "boosting", "knn", "svm", "all"]
    parser.add_argument(
        "--experiment_type",
        choices=experiment_choices,
        default=False,
        help="Type of experiment to run. Choose from: dt, ann, boosting, knn, svm, all",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="If true, provide verbose output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to set for non-determinstic portions of the experiments",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=-1,
        help="Number of threads (defaults to -1)",
    )

    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="Dry run. Only reports passed in CLI arguments",
    )
    args = parser.parse_args()

    verbose = args.verbose
    config = Config(verbose=verbose)

    if verbose:
        print(f"{parser.prog} CLI Arguments")
        print("-" * width)
        print(print_tuples(args._get_kwargs()))
        print("-" * width)

    # Collect Datasets
    datasets = [
        Dataset("winequality-white.csv", data_delimiter=";", config=config).run(
            target_col="quality"
        ),
    ]

    print("Finished Processing Dataset")
    print("-" * width)
    print("Beginning ML Experiments")

    exp_type = args.experiment_type
    if exp_type in ["dt", "all"]:
        dt_experiment_results = run_experiment_configuration(
            datasets=datasets,
            estimator=DTClassifier,
            eval_metric=DT_EVAL_METRIC,
            param_grid=DT_PARAM_GRID,
            config=config,
        )

    if exp_type in ["ann", "all"]:
        pass
    if exp_type in ["boosting", "all"]:
        pass
    if exp_type in ["knn", "all"]:
        pass
    if exp_type in ["svm", "all"]:
        pass

    print(json.dumps(dt_experiment_results, indent=4))
