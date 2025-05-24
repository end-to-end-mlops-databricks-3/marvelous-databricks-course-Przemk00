"""Custom model implementation.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.
num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

import os
import sys
from typing import Literal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import mlflow
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, Schema
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from default_detection.config import ProjectConfig, Tags
from default_detection.utils import serving_pred_function


class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    """Date features engineering class."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "DateFeatureEngineer":
        """Fit method for date feature engineering."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform method for date feature engineering."""
        X = X.copy()
        X["month_sin"] = np.sin(2 * np.pi * X["arrival_month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["arrival_month"] / 12)
        X["is_first_quarter"] = X["arrival_month"].apply(lambda x: 1 if x in [1, 2, 3] else 0)
        X["is_second_quarter"] = X["arrival_month"].apply(lambda x: 1 if x in [4, 5, 6] else 0)
        X["is_third_quarter"] = X["arrival_month"].apply(lambda x: 1 if x in [7, 8, 9] else 0)
        X["is_fourth_quarter"] = X["arrival_month"].apply(lambda x: 1 if x in [10, 11, 12] else 0)
        return X


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for machine learning models to be used with MLflow.

    This class wraps a machine learning model for predicting house prices.
    """

    def __init__(self, model: object) -> None:
        """Initialize the ModelWrapper.

        :param model: The underlying machine learning model.
        """
        self.model = model

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame:
        """Make predictions using the wrapped model.

        :param context: The MLflow context (unused in this implementation).
        :param model_input: Input data for making predictions.
        :return: A dictionary containing the adjusted prediction.
        """
        banned_client_list = pd.read_csv(context.artifacts["banned_client_list"], sep=";")
        client_ids = model_input["Client_ID"].values

        predictions = self.model.predict_proba(model_input)
        proba_canceled = predictions[:, 1]

        adjusted_predictions = serving_pred_function(client_ids, banned_client_list, proba_canceled)
        logger.info(f"adjusted_predictions: {adjusted_predictions}")

        comment = [
            "Banned" if client_id in banned_client_list["banned_clients_ids"].values else "None"
            for client_id in client_ids
        ]

        return pd.DataFrame(
            {
                "Client_ID": client_ids,
                "Proba": adjusted_predictions,
                "Comment": comment,
            }
        )


class PocessModeling:
    """Custom model class for house price prediction.

    This class encapsulates the entire workflow of loading data, preparing features,
    training the model, and making predictions.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: list[str]) -> None:
        """Initialize the PocessModeling.

        :param config: Configuration object containing model settings.
        :param tags: Tags for MLflow logging.
        :param spark: SparkSession object.
        :param code_paths: List of paths to additional code dependencies.
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.date_features = self.config.date_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_custom
        self.tags = tags.dict()
        self.code_paths = code_paths
        self.banned_clients_ids = self.config.banned_clients_ids
        self.banned_client_path = f"/Volumes/{self.catalog_name}/{self.schema_name}/alubiss/banned_client_list.csv"

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        This method loads data from Databricks tables and splits it into features and target variables.
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas().drop(columns=["update_timestamp_utc"], errors="ignore")
        self.test_set = (
            self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
            .toPandas()
            .drop(columns=["update_timestamp_utc"], errors="ignore")
        )
        self.data_version = "0"  # describe history -> retrieve

        self.X_train = self.train_set[
            self.num_features + self.cat_features + self.date_features + ["Client_ID", "Booking_ID"]
        ]
        self.y_train = self.train_set[self.target].map({"Not_Canceled": 0, "Canceled": 1})
        self.X_test = self.test_set[
            self.num_features + self.cat_features + self.date_features + ["Client_ID", "Booking_ID"]
        ]
        self.y_test = self.test_set[self.target].map({"Not_Canceled": 0, "Canceled": 1})
        self.train_set = self.train_set.drop(columns=[self.target])
        self.test_set = self.test_set.drop(columns=[self.target])

        self.banned_client_df = pd.DataFrame({"banned_clients_ids": self.banned_clients_ids})
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self) -> None:
        """Feature engineering and preprocessing."""
        logger.info("🔄 Defining preprocessing pipeline...")
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features),
                ("drop_ids", "drop", ["arrival_month", "Client_ID", "Booking_ID"]),
            ],
            remainder="passthrough",
        )

        self.pipeline = Pipeline(
            steps=[
                ("date_features", DateFeatureEngineer()),
                ("preprocessor", self.preprocessor),
                ("regressor", LGBMClassifier(**self.parameters)),
            ]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def tune_hyperparameters(self, max_evals: int = 20, n_splits: int = 3) -> None:
        """Tune hyperparameters using Hyperopt and MLflow nested runs, set best pipeline and params with CV."""
        mlflow.set_experiment(self.experiment_name)

        def objective(params: dict) -> dict:
            with mlflow.start_run(nested=True):
                f1_scores = []
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                for train_idx, valid_idx in skf.split(self.X_train, self.y_train):
                    X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[valid_idx]
                    y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[valid_idx]
                    model = LGBMClassifier(**params)
                    pipeline = Pipeline(
                        [
                            ("date_features", DateFeatureEngineer()),
                            ("preprocessor", self.preprocessor),
                            ("regressor", model),
                        ]
                    )
                    pipeline.fit(X_tr, y_tr)
                    y_pred = pipeline.predict(X_val)
                    f1 = f1_score(y_val, y_pred)
                    f1_scores.append(f1)
                mean_f1 = np.mean(f1_scores)
                mlflow.log_params(params)
                mlflow.log_metric("mean_f1_cv", mean_f1)
                return {"loss": -mean_f1, "status": STATUS_OK, "params": params, "f1": mean_f1, "pipeline": pipeline}

        space = {
            "n_estimators": hp.choice("n_estimators", [50, 100, 200]),
            "max_depth": hp.choice("max_depth", [3, 5, 7, 10]),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
            "num_leaves": hp.choice("num_leaves", [15, 31, 63, 127]),
        }

        trials = Trials()
        with mlflow.start_run(run_name="hyperopt_search", nested=True):
            fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.default_rng(42),
            )

        best_trial = sorted(trials.results, key=lambda x: -x["f1"])[0]
        logger.info(f"Best hyperparameters: {best_trial['params']}, best mean f1 (CV): {best_trial['f1']}")
        self.parameters = best_trial["params"]
        self.pipeline = Pipeline(
            [
                ("date_features", DateFeatureEngineer()),
                ("preprocessor", self.preprocessor),
                ("regressor", LGBMClassifier(**self.parameters)),
            ]
        )
        self.pipeline.fit(self.X_train, self.y_train)

    def train(self) -> None:
        """Train the model using the prepared pipeline."""
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self, dataset_type: Literal["PandasDataset", "SparkDataset"] = "SparkDataset") -> None:
        """Log the trained model and its metrics to MLflow.

        This method evaluates the model, logs parameters and metrics, and saves the model in MLflow.
        """
        mlflow.set_experiment(self.experiment_name)
        additional_pip_deps = ["pyspark==3.5.0"]
        for package in self.code_paths:
            whl_name = package.split("/")[-1]
            additional_pip_deps.append(f"./code/{whl_name}")

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            y_pred = self.pipeline.predict(self.X_test)

            # Evaluate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)

            logger.info(f"📊 Accuracy: {accuracy}")
            logger.info(f"📊 Precision: {precision}")
            logger.info(f"📊 Recall: {recall}")
            logger.info(f"📊 F1 Score: {f1}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LightGBMClassifier with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1score", f1)

            # Log the model
            if dataset_type == "PandasDataset":
                dataset = mlflow.data.from_pandas(
                    self.train_set,
                    name="train_set",
                )
            elif dataset_type == "SparkDataset":
                dataset = mlflow.data.from_spark(
                    self.train_set_spark,
                    table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                    version=self.data_version,
                )
            else:
                raise ValueError("Unsupported dataset type.")

            mlflow.log_input(dataset, context="training")

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            input_signature = Schema(
                [
                    ColSpec("integer", "required_car_parking_space"),
                    ColSpec("integer", "no_of_adults"),
                    ColSpec("integer", "no_of_children"),
                    ColSpec("integer", "no_of_weekend_nights"),
                    ColSpec("integer", "no_of_week_nights"),
                    ColSpec("integer", "lead_time"),
                    ColSpec("integer", "repeated_guest"),
                    ColSpec("integer", "no_of_previous_cancellations"),
                    ColSpec("integer", "no_of_previous_bookings_not_canceled"),
                    ColSpec("float", "avg_price_per_room"),
                    ColSpec("integer", "no_of_special_requests"),
                    ColSpec("string", "type_of_meal_plan"),
                    ColSpec("string", "room_type_reserved"),
                    ColSpec("string", "market_segment_type"),
                    ColSpec("string", "country"),
                    ColSpec("integer", "arrival_month"),
                    ColSpec("string", "Client_ID"),
                    ColSpec("string", "Booking_ID"),
                ]
            )

            output_signature = Schema(
                [
                    ColSpec("string", "Client_ID"),
                    ColSpec("double", "Proba"),
                    ColSpec("string", "Comment"),
                ]
            )

            signature = ModelSignature(inputs=input_signature, outputs=output_signature)

            mlflow.pyfunc.log_model(
                python_model=ModelWrapper(self.pipeline),
                artifact_path="pyfunc-alubiss-model",
                artifacts={"banned_client_list": self.banned_client_path},
                code_paths=self.code_paths,
                conda_env=conda_env,
                signature=signature,
                input_example=self.train_set.iloc[0:1],
            )

    def register_model(self) -> None:
        """Register the trained model in MLflow Model Registry.

        This method registers the model and sets an alias for the latest version.
        """
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/pyfunc-alubiss-model",
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_custom",
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_custom",
            alias="latest-model",
            version=latest_version,
        )

    def retrieve_current_run_dataset(self) -> DatasetSource:
        """Retrieve the dataset used in the current MLflow run.

        :return: The loaded dataset source.
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        return dataset_source.load()
        logger.info("✅ Dataset source loaded.")

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve metadata from the current MLflow run.

        :return: A tuple containing metrics and parameters of the current run.
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        return metrics, params
        logger.info("✅ Dataset metadata loaded.")

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Load the latest model (alias=latest-model) from MLflow and make predictions.

        Alias latest is not allowed -> we use latest-model instead as an alternative.
        :param input_data: Input data for prediction.
        :return: Predictions.

        Note:
        This also works
        model.unwrap_python_model().predict(None, input_data)
        check out this article:
        https://medium.com/towards-data-science/algorithm-agnostic-model-building-with-mlflow-b106a5a29535

        """
        logger.info("🔄 Loading model from MLflow alias 'production'...")

        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.hotel_reservations_model_custom@latest-model"
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info("✅ Model successfully loaded.")

        # Make predictions: None is context
        predictions = model.predict(input_data)

        # Return predictions as a DataFrame
        return predictions
