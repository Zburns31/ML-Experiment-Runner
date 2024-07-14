import pandas as pd
import numpy as np
import logging

from config import DATA_DIR, LOGS_DIR
from typing import Dict, Tuple, List
from pathlib import Path
from sklearn.model_selection import train_test_split

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', 999)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    handlers=[
        logging.FileHandler(Path(LOGS_DIR, "data_processing.log"), mode = "w+"),
        logging.StreamHandler()
    ]
)


class Dataset:
    def __init__(self, data_dir: str, dataset_name: str):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.data_path = Path(data_dir, dataset_name)
        self.data = None

    def load_data(self, verbose: bool = False, **kwargs) -> None:
        """
        Loads data from the specified path.
        """
        try:
            self.data = pd.read_csv(self.data_path, **kwargs)
            if verbose:
                logger.info(f'Loading Dataset: {self.dataset_name}')

        except FileNotFoundError:
            print(f'Dataset: {self.dataset_name} not found in location: {self.data_path}')

        logger.info("Data loaded successfully")
        logger.info(f"Number of Rows: {len(self.data)} | Number of Features: {len(self.data.columns)}")

    def summary_statistics(self, target_col: str, normalize_counts: bool = True) -> None:
        """Provides summary statistics of all columns"""
        if self.data is None or self.data.empty:
            raise ValueError("Data not loaded. Please load the data first using the load_data method")
        else:
            target_class_dist = (
                self
                .data[target_col]
                .value_counts(normalize=normalize_counts)
                .sort_index()
                .reset_index()
                .style
                .format({'proportion': '{:,.2%}'})
                .hide(axis="index")
                .to_string()
            )
            # return target_class_dist
            summary_df = self.data.describe(
                include='all',
                percentiles = [0.01, 0.25, 0.5, 0.75, 0.99]
            ).round(2)

            logger.info('Target Column: {} | Class Distribution: {}'.format(target_col, target_class_dist))
            logger.info('Data Summary: {}'.format(summary_df))


    def check_missing_values(self) -> None:
        """Checks for any missing values in the dataset."""
        has_nulls = self.data.isnull().values.any()
        if not has_nulls:
            logger.warning("Warning Missing Values Detected")

    def check_outliers(self) -> Dict[str, List[int]]:
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
                outlier_indices = self.data.index[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)].tolist()

                if outlier_indices:
                    outliers_dict[col] = outlier_indices

            logger.info(f"Outliers Detected in columns: {list(outliers_dict.keys())}")
            return outliers_dict
        else:
            raise ValueError("Data not loaded. Please load the data first using the load_data method.")

    def cast_datatypes(self, column_type_map: Dict) -> None:
        """Casts datatypes to the appropriate formats based on the data content"""
        if self.data is not None or not self.data.empty:
            pass
        else:
            raise ValueError("Data not loaded. Please load the data first using the load_data method")


    def create_train_test_split(self, target_col: str, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        has_nulls = self.data.isnull().values.any()
        if not has_nulls:
            logger.warning("Warning Missing Values Detected")

        X = self.data.loc[:, self.data.columns != target_col]
        y = self.data[target_col]

        X_TRAIN, X_TEST, y_train, y_test = train_test_split(X, y, random_state = seed, test_size = test_size)
        logger.info(f"Train Set Size: {len(X_TRAIN)} | Test Set Size: {len(X_TEST)}")

        return X_TRAIN, X_TEST, y_train, y_test



if __name__ =='__main__':

    wine_data = Dataset(DATA_DIR, 'winequality-white.csv')
    wine_data.load_data(delimiter=';')
    wine_data.summary_statistics('quality')
    wine_data.check_missing_values()
    outliers_dict = wine_data.check_outliers()

    X_TRAIN, X_TEST, y_train, y_test = wine_data.create_train_test_split('quality', 0.3, 42)

    logger.info("Finished Processing Dataset")