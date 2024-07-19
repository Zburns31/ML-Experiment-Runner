import pandas as pd
import numpy as np
import logging

from config import Config
from learners.DT import DTClassifier
from typing import Dict, Tuple, List, Type
from pathlib import Path
from sklearn.model_selection import train_test_split

pd.set_option("expand_frame_repr", False)
pd.set_option("display.max_columns", 999)

config = Config()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    handlers=[
        logging.FileHandler(Path(config.LOGS_DIR, "data_processing.log"), mode="w+"),
        logging.StreamHandler(),
    ],
)


class Dataset:
    def __init__(self, data_path: str, data_delimiter: str, config: Config):
        self.data_path: str = (
            data_path  # CHange to dataset_path and make name a property
        )
        self.data: pd.DataFrame = None
        self.delimiter: str = data_delimiter
        self.config: Type[Config] = config
        self.verbose: bool = config.VERBOSE
        self.outliers = None

    @property
    def name(self) -> str:
        return self.data_path.replace(".csv", "").replace("-", "_")

    def load_data(self, verbose: bool = False, **kwargs) -> None:
        """
        Loads data from the specified path.
        """
        try:
            data_path = Path(self.config.DATA_DIR, self.data_path)
            self.data = pd.read_csv(data_path, **kwargs)
            if verbose:
                logger.info(f"Loading Dataset: {self.name}")

        except FileNotFoundError:
            print(f"Dataset: {self.name} not found in location: {data_path}")

        logger.info("Data loaded successfully")
        logger.info(
            f"Number of Rows: {len(self.data)} | Number of Features: {len(self.data.columns)}"
        )

    def summary_statistics(self, target_col: str, normalize_counts=True) -> None:
        """Provides summary statistics of all columns"""
        if self.data is None or self.data.empty:
            raise ValueError(
                "Data not loaded. Please load the data first using the load_data method"
            )
        else:
            target_class_dist = (
                self.data[target_col]
                .value_counts(normalize=normalize_counts)
                .sort_index()
                .reset_index()
                .style.hide()  # hide index
                .format({"proportion": "{:,.2%}"})
                .to_string()
            )
            # return target_class_dist
            summary_df = self.data.describe(
                include="all", percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
            ).round(2)

            if self.verbose:
                logger.info(
                    "Target Column: {} | Class Distribution: {}".format(
                        target_col, target_class_dist
                    )
                )
                logger.info("Data Summary: {}".format(summary_df))

    def check_missing_values(self):
        """Checks for any missing values in the dataset."""
        has_nulls = self.data.isnull().values.any()
        if has_nulls:
            logger.warning("Warning Missing Values Detected")

    def check_outliers(self):
        """
        Detects outliers in each column of the DataFrame using the IQR method.

        Args:
        data (pd.DataFrame): The input DataFrame.

        Returns:
        dict: A dictionary where keys are column names and values are lists of indices of outliers.
        """
        if self.data is not None or not self.data.empty:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            outliers_dict = {}

            for col in numeric_cols:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_indices = self.data.index[
                    (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                ].tolist()

                if outlier_indices:
                    outliers_dict[col] = outlier_indices

            logger.info(f"Outliers Detected in columns: {list(outliers_dict.keys())}")
            return outliers_dict
        else:
            raise ValueError(
                "Data not loaded. Please load the data first using the load_data method."
            )

    def cast_datatypes(self, column_type_map):
        """Casts datatypes to the appropriate formats based on the data content"""
        if self.data is not None or not self.data.empty:
            pass
        else:
            raise ValueError(
                "Data not loaded. Please load the data first using the load_data method"
            )

    def create_train_test_split(
        self,
        target_col: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        X = self.data.loc[:, self.data.columns != target_col]
        y = self.data[target_col]

        stratify = y if self.config.ML_PREPROCESS_PARAMS["shuffle"] else None
        X_TRAIN, X_TEST, y_train, y_test = train_test_split(
            X,
            y,
            random_state=self.config.RANDOM_SEED,
            test_size=self.config.ML_PREPROCESS_PARAMS["test_size"],
            shuffle=self.config.ML_PREPROCESS_PARAMS["shuffle"],
            stratify=stratify,
        )
        if self.verbose:
            logger.info(
                f"Train Set Size: {len(X_TRAIN)} | Test Set Size: {len(X_TEST)}"
            )

        return X_TRAIN, X_TEST, y_train, y_test, X, y

    def scale_data(self, x_train, x_test, y_train, y_test):
        """
        TODO: Add functionality for Normalization or MinMaxScaling
        _summary_

        Args:
            x_train (_type_): _description_
            x_test (_type_): _description_
            y_train (_type_): _description_
            y_test (_type_): _description_
        """
        pass

    def run(self, target_col: str):
        self.load_data(verbose=self.verbose, delimiter=self.delimiter)
        self.summary_statistics(target_col=target_col)
        self.check_missing_values()
        self.outliers = self.check_outliers()

        X_TRAIN, X_TEST, y_train, y_test, X, y = self.create_train_test_split(
            target_col
        )
        return X_TRAIN, X_TEST, y_train, y_test, X, y


if __name__ == "__main__":

    X_TRAIN, X_TEST, y_train, y_test, X, y = Dataset(
        "winequality-white.csv", data_delimiter=";", config=config
    ).run(target_col="quality")

    logger.info("Finished Processing Dataset")

    parameter_grid = {"max_depth": np.linspace(0, 10, 1)}
    eval_metric = "accuracy"
    wine_dt = DTClassifier(config, parameter_grid, eval_metric)

    # print(wine_dt.unique_hyperparameters)
    # print(wine_dt.get_params())
    scoring_func = wine_dt.get_scorer("accuracy")

    best_param_value = wine_dt.plot_validation_curve(
        X,
        y,
        dataset_name="winequality-white",
        param_name="max_depth",
        param_range=np.arange(1, 11),
        save_plot=True,
    )

    wine_dt.set_params(max_depth=best_param_value)
    wine_dt.plot_learning_curve(
        X, y, param_name="max_depth", dataset_name="winequality-white", save_plot=True
    )
