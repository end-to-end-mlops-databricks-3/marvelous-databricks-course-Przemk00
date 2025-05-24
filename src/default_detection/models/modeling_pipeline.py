"""Baseline Default Detection model implementation.

This file provides a baseline pipeline for a default detection model.
It includes feature preprocessing, hyperparameter tuning with Hyperopt,
model training (LGBMClassifier), and MLflow integration for tracking and registration.
"""

import os  # For os.path.exists, though not strictly needed in this refactored version
from typing import Literal

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

# BaseEstimator, TransformerMixin are no longer needed as DateFeatureEngineer is removed
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from default_detection.config import ProjectConfig, Tags


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for the default detection model to be used with MLflow.

    This class wraps the trained scikit-learn pipeline.
    """

    def __init__(self, model: object) -> None:
        """Initialize the ModelWrapper.

        :param model: The underlying scikit-learn pipeline.
        """
        self.model = model

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the wrapped model.

        :param context: The MLflow context (unused in this implementation).
        :param model_input: Input DataFrame with features for making predictions.
        :return: A DataFrame containing the raw prediction and probability of default.
        """
        raw_predictions = self.model.predict(model_input)
        probabilities = self.model.predict_proba(model_input)
        # Assuming class 1 is the "default" (positive) class
        probability_default = probabilities[:, 1]

        return pd.DataFrame(
            {
                "prediction": raw_predictions,
                "probability_default": probability_default,
            }
        )


class PocessModeling:
    """Baseline model class for default detection.

    This class encapsulates the workflow of loading data, preparing features,
    training a LightGBM model, and logging it with MLflow.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: list[str]) -> None:
        """Initialize the DefaultDetectionBaselineModeling.

        :param config: ProjectConfig object with num_features, cat_features, target, parameters, etc.
        :param tags: Tags for MLflow logging.
        :param spark: SparkSession object.
        :param code_paths: List of paths to additional code dependencies for MLflow.
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name
        self.tags = tags.dict()
        self.code_paths = code_paths

    def load_data(self) -> None:
        """Load training and testing data from Delta tables using Spark.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables for default detection baseline...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas().drop(columns=["update_timestamp_utc"], errors="ignore")
        self.test_set = (
            self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
            .toPandas()
            .drop(columns=["update_timestamp_utc"], errors="ignore")
        )

        try:  # Attempt to get version from Spark table history
            self.data_version = str(
                self.train_set_spark.history().select("version").orderBy("version", ascending=False).first()[0]
            )
        except Exception:
            logger.warning("Could not retrieve Spark table version. Defaulting data_version to '0'.")
            self.data_version = "0"

        # Features for training are only num_features and cat_features
        self.X_train = self.train_set[self.num_features + self.cat_features]
        # Assuming target 'Y' is already 0 or 1. No mapping needed.
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]

        # For MLflow dataset logging, we'll log the feature DataFrames
        self.train_set_features_for_logging = self.X_train.copy()

        logger.info("✅ Data successfully loaded for default detection baseline.")

    def prepare_features(self) -> None:
        """Prepare the feature preprocessing pipeline."""
        logger.info("🔄 Defining preprocessing pipeline for default detection baseline...")
        # Preprocessor for categorical features (OneHotEncoding)
        # Numerical features will be passed through by 'remainder="passthrough"'
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_features),
            ],
            remainder="passthrough",  # Passes through num_features
        )

        # Full scikit-learn pipeline
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", LGBMClassifier(**self.parameters)),  # Changed from regressor
            ]
        )
        logger.info("✅ Preprocessing pipeline defined for default detection baseline.")

    def tune_hyperparameters(self, max_evals: int = 20, n_splits: int = 3) -> None:
        """Tune hyperparameters using Hyperopt and MLflow nested runs."""
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"🔄 Starting hyperparameter tuning for '{self.experiment_name}'...")

        # Use only feature columns for hyperparameter tuning
        X_train_hyperopt = self.X_train.copy()
        y_train_hyperopt = self.y_train.copy()

        def objective(params: dict) -> dict:
            with mlflow.start_run(nested=True, run_name="hyperopt_trial"):
                f1_scores_cv = []
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                for train_idx, valid_idx in skf.split(X_train_hyperopt, y_train_hyperopt):
                    X_tr, X_val = X_train_hyperopt.iloc[train_idx], X_train_hyperopt.iloc[valid_idx]
                    y_tr, y_val = y_train_hyperopt.iloc[train_idx], y_train_hyperopt.iloc[valid_idx]

                    model_trial = LGBMClassifier(**params)
                    # Pipeline for this trial
                    trial_pipeline = Pipeline(
                        steps=[
                            ("preprocessor", self.preprocessor),  # Use the class's preprocessor
                            ("classifier", model_trial),
                        ]
                    )
                    trial_pipeline.fit(X_tr, y_tr)
                    y_pred_val = trial_pipeline.predict(X_val)
                    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
                    f1_scores_cv.append(f1_val)

                mean_f1_cv = np.mean(f1_scores_cv)
                mlflow.log_params(params)
                mlflow.log_metric("mean_f1_cv", mean_f1_cv)
                # Return loss (negative F1 for maximization) and other info
                return {"loss": -mean_f1_cv, "status": STATUS_OK, "params": params, "f1_cv": mean_f1_cv}

        # Define hyperparameter search space (example)
        space = {
            "n_estimators": hp.choice("n_estimators", self.parameters.get("n_estimators_space", [100, 200, 300])),
            "max_depth": hp.choice("max_depth", self.parameters.get("max_depth_space", [-1, 5, 10])),
            "learning_rate": hp.uniform(
                "learning_rate",
                self.parameters.get("learning_rate_min", 0.01),
                self.parameters.get("learning_rate_max", 0.1),
            ),
            "num_leaves": hp.choice("num_leaves", self.parameters.get("num_leaves_space", [31, 50, 70])),
            "random_state": 42,  # For reproducibility
            "class_weight": "balanced",  # Good for imbalanced datasets
        }
        # Allow fixed parameters from config if not in space, or override space with config if specific values given
        # For simplicity, this example uses defaults or ranges from self.parameters if *_space/*_min/*_max keys exist

        trials = Trials()
        with mlflow.start_run(run_name="hyperopt_search_model", nested=True, tags=self.tags):
            fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=np.random.default_rng(42),  # For reproducibility of Hyperopt suggestions
            )

        # Get the best trial (highest F1 score)
        best_trial_results = sorted(trials.results, key=lambda x: x["f1_cv"], reverse=True)
        if not best_trial_results:
            logger.error("Hyperopt tuning did not produce any results. Using initial parameters.")
        else:
            best_trial = best_trial_results[0]
            logger.info(
                f"✅ Best hyperparameters found: {best_trial['params']}, Best Mean F1 (CV): {best_trial['f1_cv']:.4f}"
            )
            self.parameters.update(best_trial["params"])  # Update self.parameters with best found

        # Re-initialize the main pipeline with the best (or initial if tuning failed) parameters
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", LGBMClassifier(**self.parameters)),
            ]
        )
        # Fit the final pipeline on the full training data
        self.pipeline.fit(self.X_train, self.y_train)
        logger.info("✅ Hyperparameter tuning complete and final pipeline fitted.")

    def train(self) -> None:
        """Train the model using the prepared pipeline (after potential hyperparameter tuning)."""
        logger.info("🚀 Starting training for default detection baseline model...")
        # Assuming self.pipeline is already defined and fitted in tune_hyperparameters or prepare_features
        if not hasattr(self, "pipeline"):
            logger.warning("Pipeline not initialized. Running prepare_features and fitting.")
            self.prepare_features()  # Ensure pipeline is created
            self.pipeline.fit(self.X_train, self.y_train)
        elif not getattr(self.pipeline, "_is_fitted", False):  # Check if pipeline is fitted
            self.pipeline.fit(self.X_train, self.y_train)
        logger.info("✅ Training completed for default detection baseline model.")

    def log_model(self, dataset_type: Literal["PandasDataset", "SparkDataset"] = "SparkDataset") -> None:
        """Log the trained model and its metrics to MLflow."""
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"📦 Logging model to MLflow experiment: '{self.experiment_name}'")

        additional_pip_deps = ["pyspark==3.5.0"]
        if self.code_paths:
            for package in self.code_paths:
                whl_name = os.path.basename(package)  # Use os.path.basename
                additional_pip_deps.append(f"./code/{whl_name}")

        with mlflow.start_run(tags=self.tags, run_name="log_model") as run:
            self.run_id = run.info.run_id
            y_pred_test = self.pipeline.predict(self.X_test)

            # Evaluate metrics
            accuracy = accuracy_score(self.y_test, y_pred_test)
            precision = precision_score(self.y_test, y_pred_test, zero_division=0)
            recall = recall_score(self.y_test, y_pred_test, zero_division=0)
            f1 = f1_score(self.y_test, y_pred_test, zero_division=0)

            logger.info("📊 Test Set Metrics for Baseline Model:")
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1 Score:  {f1:.4f}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LGBMClassifier_baseline_default_detection")
            mlflow.log_params(self.parameters)  # Log final parameters used
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1_score", f1)

            # Log the training dataset (features only)
            if dataset_type == "PandasDataset":
                dataset_log = mlflow.data.from_pandas(
                    self.train_set_features_for_logging,  # This is X_train
                    name="train_set_features_default_detection_baseline",
                )
            elif dataset_type == "SparkDataset":
                # Create Spark DataFrame from X_train for logging if needed, or use original train_set_spark with select
                train_set_spark_features = self.train_set_spark.select(self.num_features + self.cat_features)
                dataset_log = mlflow.data.from_spark(
                    train_set_spark_features,
                    name=f"{self.catalog_name}.{self.schema_name}.train_set_features_baseline",
                    version=self.data_version,  # Use the determined data version
                )
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
            mlflow.log_input(dataset_log, context="training_features")

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            # Define input_signature based on project_config.yml features
            input_schema_cols = []
            # Assuming types from project_config.yml comments:
            # X1 (LIMIT_BAL) - int
            # X5 (AGE) - int
            # X6-X11 (PAY_0 to PAY_6) - int
            # X12-X17 (BILL_AMT1 to BILL_AMT6) - float (double)
            # X18-X23 (PAY_AMT1 to PAY_AMT6) - float (double)
            int_num_features_config = ["X1", "X5", "X6", "X7", "X8", "X9", "X10", "X11"]
            # All other num_features are assumed float/double

            for feature in self.config.num_features:
                if feature in int_num_features_config:
                    input_schema_cols.append(ColSpec("integer", feature))
                else:  # Default to double for other numerical features
                    input_schema_cols.append(ColSpec("double", feature))

            # X2 (Sex) - string (cat)
            # X3 (EDUCATION) - string (cat)
            # X4 (MARRIAGE) - string (cat)
            for feature in self.config.cat_features:
                input_schema_cols.append(ColSpec("string", feature))

            input_signature = Schema(input_schema_cols)

            # Define output_signature for the simplified ModelWrapper
            output_signature = Schema(
                [
                    ColSpec("integer", "prediction"),  # Raw model prediction (0 or 1)
                    ColSpec("double", "probability_default"),  # Probability of class 1
                ]
            )
            signature = ModelSignature(inputs=input_signature, outputs=output_signature)

            # Prepare input example using only feature columns from X_train
            input_example_df = self.X_train.iloc[0:1].copy()

            mlflow.pyfunc.log_model(
                python_model=ModelWrapper(self.pipeline),  # Pass the scikit-learn pipeline
                artifact_path="pyfunc_default_detection_model",  # New artifact path
                # artifacts dictionary removed as no extra artifacts for this baseline
                code_paths=self.code_paths,
                conda_env=conda_env,
                signature=signature,
                input_example=input_example_df,
            )
            logger.info(f"✅ Baseline model logged with run_id: {self.run_id}")

    def register_model(self) -> None:
        """Register the trained baseline model in MLflow Model Registry."""
        if not hasattr(self, "run_id") or not self.run_id:
            logger.error("Run ID not found. Please run log_model first before registering.")
            return

        logger.info("🔄 Registering the baseline model in UC...")
        model_name = f"{self.catalog_name}.{self.schema_name}.default_detection_model"  # New model name

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/pyfunc_default_detection_model",  # Use new artifact path
            name=model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Baseline model registered as '{model_name}' version {registered_model.version}.")

        latest_version = registered_model.version
        client = MlflowClient()
        alias_name = "Baseline"  # Example alias for this model
        try:
            client.set_registered_model_alias(
                name=model_name,
                alias=alias_name,
                version=latest_version,
            )
            logger.info(f"✅ Alias '{alias_name}' set for version {latest_version} of model '{model_name}'.")
        except Exception as e:
            logger.error(f"Failed to set alias '{alias_name}' for model '{model_name}': {e}")

    def retrieve_current_run_dataset(self) -> DatasetSource | None:  # type: ignore[type-arg]
        """Retrieve the dataset used in the current MLflow run."""
        if not hasattr(self, "run_id") or not self.run_id:
            logger.error("Run ID not found. Please run log_model first.")
            return None

        run = mlflow.get_run(self.run_id)
        if not run.inputs.dataset_inputs:
            logger.error("No dataset inputs found for the run.")
            return None

        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)  # type: ignore[attr-defined]
        loaded_data = dataset_source.load()
        logger.info("✅ Dataset source loaded for current run.")
        return loaded_data

    def retrieve_current_run_metadata(self) -> tuple[dict, dict] | None:
        """Retrieve metadata (metrics and params) from the current MLflow run."""
        if not hasattr(self, "run_id") or not self.run_id:
            logger.error("Run ID not found. Please run log_model first.")
            return None

        run = mlflow.get_run(self.run_id)
        metrics = run.data.metrics
        params = run.data.params
        logger.info("✅ Run metadata (metrics, params) loaded for current run.")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame, alias: str = "Baseline") -> pd.DataFrame | None:
        """Load the latest baseline model (by alias) from MLflow and make predictions.

        :param input_data: Input DataFrame with features for prediction.
        :param alias: The model alias to load (e.g., "Baseline").
        :return: Predictions as a DataFrame (prediction, probability_default), or None if error.
        """
        model_name = f"{self.catalog_name}.{self.schema_name}.default_detection_model"  # Use new model name
        logger.info(f"🔄 Loading baseline model '{model_name}@{alias}' from MLflow...")

        try:
            model_uri = f"models:/{model_name}@{alias}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("✅ Baseline model successfully loaded.")

            # ModelWrapper expects a DataFrame with feature columns
            predictions_df = model.predict(input_data)
            return predictions_df
        except Exception as e:
            logger.error(f"Error loading baseline model or making predictions: {e}")
            return None
