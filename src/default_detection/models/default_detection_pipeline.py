"""Default detection model implementation.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.
num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

from typing import Literal

import mlflow
import numpy as np
import pandas as pd

# Removed: from hotel_reservations.utils import serving_pred_function
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, Schema
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from default_detection.config import ProjectConfig, Tags  # Changed from hotel_reservations

# Removed DateFeatureEngineer class


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for machine learning models to be used with MLflow.

    This class wraps a machine learning model for default detection.
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
        :return: A DataFrame containing the prediction and probability of default.
        """
        # Removed banned_client_list logic
        # Removed client_ids extraction for banned list

        raw_predictions = self.model.predict(model_input)
        probabilities = self.model.predict_proba(model_input)
        # Assuming class 1 is the "default" class
        probability_default = probabilities[:, 1]

        # Removed comment generation logic

        return pd.DataFrame(
            {
                "prediction": raw_predictions,
                "probability_default": probability_default,
            }
        )


class DefaultDetectionModeling:  # Renamed from PocessModeling
    """Custom model class for default detection.

    This class encapsulates the entire workflow of loading data, preparing features,
    training the model, and making predictions.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: list[str]) -> None:
        """Initialize the DefaultDetectionModeling.

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
        # Removed: self.date_features = self.config.date_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        # Changed experiment name
        self.experiment_name = (
            f"{self.config.schema_name}_Default_Detection_Experiment"  # Using schema_name for uniqueness
        )
        self.tags = tags.dict()
        self.code_paths = code_paths
        # Removed: self.banned_clients_ids = self.config.banned_clients_ids
        # Removed: self.banned_client_path = f"/Volumes/{self.catalog_name}/{self.schema_name}/alubiss/banned_client_list.csv"

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
        # TODO: Consider how to get data_version if needed, for now hardcoding as in original
        self.data_version = self.train_set_spark.version() if hasattr(self.train_set_spark, "version") else "0"

        # Adjusted feature selection, removed date_features and specific IDs like Client_ID, Booking_ID
        self.X_train = self.train_set[self.num_features + self.cat_features]
        # Assuming target 'Y' is already 0 or 1. If not, mapping might be needed.
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]

        # The following lines drop the target from the full train/test sets if they are to be logged
        # or used elsewhere without the target.
        self.train_set_for_logging = self.train_set.drop(columns=[self.target], errors="ignore")
        self.test_set_for_logging = self.test_set.drop(columns=[self.target], errors="ignore")

        # Removed: self.banned_client_df = pd.DataFrame({"banned_clients_ids": self.banned_clients_ids})
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self) -> None:
        """Feature engineering and preprocessing."""
        logger.info("🔄 Defining preprocessing pipeline...")
        # Simplified preprocessor: only OneHotEncoder for categorical features.
        # Numerical features will be passed through by 'remainder="passthrough"'.
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_features),
            ],
            remainder="passthrough",
        )

        self.pipeline = Pipeline(
            steps=[
                # Removed: ("date_features", DateFeatureEngineer()),
                ("preprocessor", self.preprocessor),
                ("classifier", LGBMClassifier(**self.parameters)),  # Renamed from regressor to classifier
            ]
        )
        logger.info("✅ Preprocessing pipeline defined.")

    def tune_hyperparameters(self, max_evals: int = 20, n_splits: int = 3) -> None:
        """Tune hyperparameters using Hyperopt and MLflow nested runs, set best pipeline and params with CV."""
        mlflow.set_experiment(self.experiment_name)

        # Make sure X_train_hyperopt contains only features, not other IDs if any were kept temporarily
        X_train_hyperopt = self.X_train[self.num_features + self.cat_features]
        y_train_hyperopt = self.y_train

        def objective(params: dict) -> dict:
            with mlflow.start_run(nested=True):
                f1_scores = []
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                for train_idx, valid_idx in skf.split(X_train_hyperopt, y_train_hyperopt):
                    X_tr, X_val = X_train_hyperopt.iloc[train_idx], X_train_hyperopt.iloc[valid_idx]
                    y_tr, y_val = y_train_hyperopt.iloc[train_idx], y_train_hyperopt.iloc[valid_idx]

                    model = LGBMClassifier(**params)
                    # Define pipeline for this specific trial run
                    current_pipeline = Pipeline(
                        [
                            # Removed: ("date_features", DateFeatureEngineer()),
                            ("preprocessor", self.preprocessor),  # Use the class's preprocessor
                            ("classifier", model),  # Renamed
                        ]
                    )
                    current_pipeline.fit(X_tr, y_tr)
                    y_pred = current_pipeline.predict(X_val)
                    f1 = f1_score(y_val, y_pred)
                    f1_scores.append(f1)
                mean_f1 = np.mean(f1_scores)
                mlflow.log_params(params)
                mlflow.log_metric("mean_f1_cv", mean_f1)
                # Returning the pipeline object from objective can be memory intensive with many trials.
                # It's better to just return params and score.
                return {"loss": -mean_f1, "status": STATUS_OK, "params": params, "f1": mean_f1}

        space = {
            "n_estimators": hp.choice("n_estimators", [50, 100, 200]),  # Example space, adjust as needed
            "max_depth": hp.choice("max_depth", [3, 5, 7, 10]),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
            "num_leaves": hp.choice("num_leaves", [15, 31, 63, 127]),
            "random_state": 42,  # Added for reproducibility
            "class_weight": "balanced",  # Often useful for imbalanced classification
        }
        # Update self.parameters with any fixed ones from config, allowing hyperopt to override search space
        # This assumes self.parameters from config are the base, and space defines search range
        # For simplicity, the space above is self-contained. If config.parameters should be used as fixed,
        # then they should not be in `space`.

        trials = Trials()
        with mlflow.start_run(run_name="hyperopt_search_default_detection", nested=True, tags=self.tags):
            fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.default_rng(42),
            )

        best_trial = sorted(trials.results, key=lambda x: x["f1"], reverse=True)[0]  # Sort by f1 descending
        logger.info(f"Best hyperparameters: {best_trial['params']}, best mean f1 (CV): {best_trial['f1']}")
        self.parameters = best_trial["params"]  # Update class parameters with best found

        # Re-initialize the main pipeline with best parameters
        self.pipeline = Pipeline(
            [
                # Removed: ("date_features", DateFeatureEngineer()),
                ("preprocessor", self.preprocessor),
                ("classifier", LGBMClassifier(**self.parameters)),  # Renamed
            ]
        )
        self.pipeline.fit(self.X_train, self.y_train)  # Fit with best params on full training data

    def train(self) -> None:
        """Train the model using the prepared pipeline."""
        logger.info("🚀 Starting training...")
        # Ensure X_train and y_train are correctly prepared before this step
        self.pipeline.fit(self.X_train, self.y_train)
        logger.info("✅ Training completed.")

    def log_model(self, dataset_type: Literal["PandasDataset", "SparkDataset"] = "SparkDataset") -> None:
        """Log the trained model and its metrics to MLflow.

        This method evaluates the model, logs parameters and metrics, and saves the model in MLflow.
        """
        mlflow.set_experiment(self.experiment_name)
        additional_pip_deps = ["pyspark==3.5.0"]  # Keep or adjust as needed
        if self.code_paths:
            for package in self.code_paths:
                whl_name = package.split("/")[-1]
                additional_pip_deps.append(f"./code/{whl_name}")

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            y_pred = self.pipeline.predict(self.X_test)

            # Evaluate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)  # Add zero_division_behavior if needed
            recall = recall_score(self.y_test, y_pred)  # Add zero_division_behavior if needed
            f1 = f1_score(self.y_test, y_pred)  # Add zero_division_behavior if needed

            logger.info(f"📊 Accuracy: {accuracy}")
            logger.info(f"📊 Precision: {precision}")
            logger.info(f"📊 Recall: {recall}")
            logger.info(f"📊 F1 Score: {f1}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LGBMClassifier with preprocessing")
            mlflow.log_params(self.parameters)  # Log the final parameters used for training
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1score", f1)

            # Log the dataset
            if dataset_type == "PandasDataset":
                # Use train_set_for_logging which doesn't have the target column if that's desired for input schema
                dataset = mlflow.data.from_pandas(
                    self.train_set_for_logging[self.num_features + self.cat_features],  # Log only features
                    name="train_set_features_default_detection",
                )
            elif dataset_type == "SparkDataset":
                # Assuming train_set_spark is the Spark DataFrame of the training data
                # Select only feature columns for the dataset to be logged if schema is strict
                train_set_spark_features = self.train_set_spark.select(self.num_features + self.cat_features)
                dataset = mlflow.data.from_spark(
                    train_set_spark_features,
                    name=f"{self.catalog_name}.{self.schema_name}.train_set_features_default_detection",
                    # table_name=f"{self.catalog_name}.{self.schema_name}.train_set", # Original table name
                    version=self.data_version,
                )
            else:
                raise ValueError("Unsupported dataset type.")

            mlflow.log_input(dataset, context="training")

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            # Define input_signature based on project_config.yml
            # X1 (LIMIT_BAL) - integer
            # X2 (Sex) - string
            # X3 (EDUCATION) - string
            # X4 (MARRIAGE) - string
            # X5 (AGE) - integer
            # X6-X11 (PAY_0 to PAY_6) - integer
            # X12-X17 (BILL_AMT1 to BILL_AMT6) - float
            # X18-X23 (PAY_AMT1 to PAY_AMT6) - float
            input_schema_cols = []
            # Numerical features from project_config.yml
            # X1, X5, X6, X7, X8, X9, X10, X11
            int_num_features = ["X1", "X5", "X6", "X7", "X8", "X9", "X10", "X11"]
            # X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23
            float_num_features = ["X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23"]

            for feature in self.config.num_features:
                if feature in int_num_features:
                    input_schema_cols.append(ColSpec("integer", feature))
                elif feature in float_num_features:
                    input_schema_cols.append(ColSpec("double", feature))  # MLflow uses double for float
                else:  # Default to double if not specified, or handle error
                    input_schema_cols.append(ColSpec("double", feature))

            # Categorical features from project_config.yml
            # X2, X3, X4
            for feature in self.config.cat_features:
                input_schema_cols.append(ColSpec("string", feature))

            input_signature = Schema(input_schema_cols)

            # Define output_signature for ModelWrapper
            output_signature = Schema(
                [
                    ColSpec("integer", "prediction"),
                    ColSpec("double", "probability_default"),
                ]
            )

            signature = ModelSignature(inputs=input_signature, outputs=output_signature)

            # Prepare input example using only feature columns
            input_example_df = self.X_train[self.num_features + self.cat_features].iloc[0:1]

            mlflow.pyfunc.log_model(
                python_model=ModelWrapper(self.pipeline),
                artifact_path="pyfunc-default-detection-model",  # Changed artifact path
                # Removed: artifacts={"banned_client_list": self.banned_client_path},
                code_paths=self.code_paths,
                conda_env=conda_env,
                signature=signature,
                input_example=input_example_df,  # Use the correctly shaped input example
            )
            logger.info(f"✅ Model logged with run_id: {self.run_id}")

    def register_model(self) -> None:
        """Register the trained model in MLflow Model Registry.

        This method registers the model and sets an alias for the latest version.
        """
        logger.info("🔄 Registering the model in UC...")
        model_name = f"{self.catalog_name}.{self.schema_name}.default_detection_model"  # Changed model name
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/pyfunc-default-detection-model",  # Changed artifact path
            name=model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as '{model_name}' version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        # Consider making alias configurable or more generic
        alias_name = "Challenger"  # Example alias
        client.set_registered_model_alias(
            name=model_name,
            alias=alias_name,
            version=latest_version,
        )
        logger.info(f"✅ Alias '{alias_name}' set for version {latest_version} of model '{model_name}'.")

    def retrieve_current_run_dataset(self) -> DatasetSource:  # type: ignore[type-arg]
        """Retrieve the dataset used in the current MLflow run.

        :return: The loaded dataset source.
        """
        if not hasattr(self, "run_id") or not self.run_id:
            logger.error("Run ID not found. Please run log_model first.")
            return None  # type: ignore

        run = mlflow.get_run(self.run_id)
        if not run.inputs.dataset_inputs:
            logger.error("No dataset inputs found for the run.")
            return None  # type: ignore

        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)  # type: ignore[attr-defined]
        loaded_data = dataset_source.load()
        logger.info("✅ Dataset source loaded.")
        return loaded_data

    def retrieve_current_run_metadata(self) -> tuple[dict, dict] | None:
        """Retrieve metadata from the current MLflow run.

        :return: A tuple containing metrics and parameters of the current run, or None if error.
        """
        if not hasattr(self, "run_id") or not self.run_id:
            logger.error("Run ID not found. Please run log_model first.")
            return None

        run = mlflow.get_run(self.run_id)
        metrics = run.data.metrics
        params = run.data.params
        logger.info("✅ Run metadata (metrics, params) loaded.")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame, alias: str = "Challenger") -> pd.DataFrame | None:
        """Load the latest model (by alias) from MLflow and make predictions.

        :param input_data: Input data for prediction.
        :param alias: The model alias to load (e.g., "Challenger", "Champion").
        :return: Predictions as a DataFrame, or None if error.
        """
        model_name = f"{self.catalog_name}.{self.schema_name}.default_detection_model"
        logger.info(f"🔄 Loading model '{model_name}@{alias}' from MLflow...")

        try:
            model_uri = f"models:/{model_name}@{alias}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("✅ Model successfully loaded.")

            predictions = model.predict(input_data)
            return predictions  # This will be a DataFrame as per ModelWrapper's output
        except Exception as e:
            logger.error(f"Error loading model or making predictions: {e}")
            return None
