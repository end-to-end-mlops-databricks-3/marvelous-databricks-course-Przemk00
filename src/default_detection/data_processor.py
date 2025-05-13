"""Data preprocessing module."""

import datetime
import logging

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from default_detection.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df.copy()  # Work on a copy to avoid modifying original DataFrame outside class
        self.config = config
        self.spark = spark
        self.logger = logging.getLogger(__name__)

    def preprocess(self) -> None:
        """Preprocess the DataFrame for credit card default prediction.

        Handles data type conversions and feature selection based on ProjectConfig.
        Assumes no missing values as per the provided dataset description.
        """
        self.logger.info(f"Starting preprocessing. Initial DataFrame shape: {self.df.shape}")
        # Ensure ID is string if present and used as an identifier
        # Based on schema, 'ID' is the identifier column name.
        if "ID" in self.df.columns:
            self.df["ID"] = self.df["ID"].astype(str)
            self.logger.info("Converted 'ID' column to string.")

        # Convert specified categorical features to 'category' dtype
        # Assumes self.config.cat_features = ["X2", "X3", "X4"] or similar from YAML
        self.logger.info(f"Attempting to convert categorical features: {self.config.cat_features}")
        for cat_col in self.config.cat_features:
            if cat_col in self.df.columns:
                self.df[cat_col] = self.df[cat_col].astype("category")
                self.logger.info(f"Converted '{cat_col}' to category.")
            else:
                self.logger.warning(f"Configured categorical feature '{cat_col}' not found in DataFrame.")

        # Ensure numeric features are numeric
        # Assumes self.config.num_features lists all X features intended as numeric.
        self.logger.info(f"Attempting to convert numeric features: {self.config.num_features}")
        for num_col in self.config.num_features:
            if num_col in self.df.columns:
                self.df[num_col] = pd.to_numeric(self.df[num_col], errors='coerce')
                if self.df[num_col].isnull().any(): # Should not happen based on schema
                    self.logger.warning(f"NaNs introduced in numeric column '{num_col}' after to_numeric. Dataset schema indicated no missing values.")
                self.logger.info(f"Converted '{num_col}' to numeric.")
            else:
                self.logger.warning(f"Configured numeric feature '{num_col}' not found in DataFrame.")
        
        # Ensure target variable is numeric (binary 0/1)
        # Assumes self.config.target = "Y"
        if self.config.target in self.df.columns:
            self.df[self.config.target] = pd.to_numeric(self.df[self.config.target], errors='raise')
            self.logger.info(f"Converted target column '{self.config.target}' to numeric.")
        else:
            raise ValueError(f"Target column '{self.config.target}' as configured was not found in DataFrame.")

        # Select only relevant columns as defined in config + ID (if present) + target
        all_configured_features = self.config.num_features + self.config.cat_features
        features_present_in_df = [col for col in all_configured_features if col in self.df.columns]
        
        columns_to_keep = features_present_in_df + [self.config.target]
        # Add ID to columns_to_keep if it exists in df and is not already part of features/target
        if "ID" in self.df.columns and "ID" not in columns_to_keep:
            columns_to_keep.append("ID")
        
        missing_configured_features = set(all_configured_features) - set(features_present_in_df)
        if missing_configured_features:
            self.logger.warning(f"The following configured features are not in the DataFrame and will be ignored for final selection: {missing_configured_features}")

        # Ensure all columns_to_keep actually exist in the df before selection
        final_columns_to_keep = [col for col in columns_to_keep if col in self.df.columns]
        self.df = self.df[final_columns_to_keep]
        
        self.logger.info(f"Preprocessing for credit card default data complete. Final DataFrame shape: {self.df.shape}. Columns: {self.df.columns.tolist()}")

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )
        self.logger.info(f"Saved train and test sets to {self.config.catalog_name}.{self.config.schema_name}")


    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
        self.logger.info(f"Enabled Change Data Feed for tables in {self.config.catalog_name}.{self.config.schema_name}")