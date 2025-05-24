"""Default detection model implementation.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.
num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

import os  # Added for os.path.exists
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
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from default_detection.config import ProjectConfig, Tags

# Import the new utility function
from default_detection.utils import adjust_probabilities_for_high_risk

# Removed DateFeatureEngineer class
# Removed unused sklearn.base imports


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class for machine learning models to be used with MLflow.

    This class wraps a machine learning model for default detection and applies
    custom logic for high-risk segment probability adjustment.
    """

    def __init__(
        self,
        model: object,
        feature_cols: list[str],
        client_id_col_name: str,
        high_risk_artifact_name_in_mlflow: str,
        client_id_col_in_artifact: str,
        probability_adjustment_factor: float,
        probability_cap_value: float,
    ) -> None:
        """Initialize the ModelWrapper.

        :param model: The underlying machine learning model (scikit-learn pipeline).
        :param feature_cols: List of feature column names for the model.
        :param client_id_col_name: Name of the client ID column in the input DataFrame.
        :param high_risk_artifact_name_in_mlflow: Key for the high-risk artifact in MLflow context.
        :param client_id_col_in_artifact: Name of the client ID column in the high-risk artifact.
        :param probability_adjustment_factor: Factor to adjust probability for high-risk clients.
        :param probability_cap_value: Maximum probability after adjustment.
        """
        self.model = model
        self.feature_cols = feature_cols
        self.client_id_col_name = client_id_col_name
        self.high_risk_artifact_name_in_mlflow = high_risk_artifact_name_in_mlflow
        self.client_id_col_in_artifact = client_id_col_in_artifact
        self.probability_adjustment_factor = probability_adjustment_factor
        self.probability_cap_value = probability_cap_value

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the wrapped model and apply high-risk adjustments.

        :param context: The MLflow context, used to load artifacts.
        :param model_input: Input DataFrame containing client_id and feature columns.
        :return: A DataFrame with client_id, adjusted probability, and final prediction.
        """
        if self.client_id_col_name not in model_input.columns:
            raise ValueError(
                f"Client ID column '{self.client_id_col_name}' not found in model_input. "
                f"Columns: {model_input.columns.tolist()}"
            )

        # Ensure all feature columns are present
        missing_features = [col for col in self.feature_cols if col not in model_input.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns in model_input: {missing_features}")

        model_features_input = model_input[self.feature_cols]

        # Get initial probabilities from the core model
        # Assuming class 1 is the "default" (positive) class
        raw_probabilities_all_classes = self.model.predict_proba(model_features_input)
        probability_default_raw = raw_probabilities_all_classes[:, 1]

        # Load high-risk segment artifact
        try:
            high_risk_df = pd.read_csv(context.artifacts[self.high_risk_artifact_name_in_mlflow])
        except Exception as e:
            logger.error(f"Failed to load high-risk artifact '{self.high_risk_artifact_name_in_mlflow}': {e}")
            # Fallback: use raw probabilities if artifact loading fails
            adjusted_probability_default = probability_default_raw
        else:
            # Apply high-risk adjustment
            adjusted_probability_default = adjust_probabilities_for_high_risk(
                model_input_df=model_input,  # Pass the full input_df for client_id lookup
                probabilities=probability_default_raw,
                high_risk_artifact_df=high_risk_df,
                client_id_col_in_input=self.client_id_col_name,
                client_id_col_in_artifact=self.client_id_col_in_artifact,
                adjustment_factor=self.probability_adjustment_factor,
                cap_value=self.probability_cap_value,
            )

        # Determine final prediction based on adjusted probability (e.g., threshold 0.5)
        final_prediction = (adjusted_probability_default >= 0.5).astype(int)

        return pd.DataFrame(
            {
                self.client_id_col_name: model_input[self.client_id_col_name],
                "adjusted_probability_default": adjusted_probability_default,
                "final_prediction": final_prediction,
            }
        )


class DefaultDetectionModeling:  # Renamed from PocessModeling
    """Custom model class for default detection.

    This class encapsulates the entire workflow of loading data, preparing features,
    training the model, and making predictions, including high-risk segment adjustments.
    """

    def __init__(
        self,
        config: ProjectConfig,
        tags: Tags,
        spark: SparkSession,
        code_paths: list[str],
        client_id_col_name: str = "ID",  # Added
        high_risk_artifact_path_for_logging: str | None = None,  # Added: e.g., "/Volumes/cat/schem/vol/high_risk.csv"
        client_id_col_in_artifact: str = "client_identifier",  # Added
        probability_adjustment_factor: float = 0.15,  # Added
        probability_cap_value: float = 0.99,  # Added
        high_risk_artifact_name_in_mlflow: str = "high_risk_segment_data",  # Added
    ) -> None:
        """Initialize the DefaultDetectionModeling.

        # ... (rest of docstring)
        :param client_id_col_name: Name of the client ID column in the main dataset.
        :param high_risk_artifact_path_for_logging: Filesystem path to the high-risk client CSV for MLflow to log.
        :param client_id_col_in_artifact: Name of the client ID column in the high-risk artifact.
        :param probability_adjustment_factor: Factor to adjust probability for high-risk clients.
        :param probability_cap_value: Maximum probability after adjustment.
        :param high_risk_artifact_name_in_mlflow: Key for the artifact in MLflow context (ModelWrapper).
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
        self.experiment_name = f"{self.config.schema_name}_Default_Detection_Experiment"
        self.tags = tags.dict()
        self.code_paths = code_paths

        # New parameters for high-risk adjustment
        self.client_id_col_name = client_id_col_name
        self.high_risk_artifact_path_for_logging = high_risk_artifact_path_for_logging
        self.client_id_col_in_artifact = client_id_col_in_artifact
        self.probability_adjustment_factor = probability_adjustment_factor
        self.probability_cap_value = probability_cap_value
        self.high_risk_artifact_name_in_mlflow = high_risk_artifact_name_in_mlflow

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Ensures client_id_col_name is present in pandas DataFrames for later use.
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        # Ensure client_id_col_name is selected if it's not part of num_features or cat_features
        # For simplicity, assuming it's loaded by default from the table.
        # If not, the spark.table select statement would need to include it.
        self.train_set = self.train_set_spark.toPandas().drop(columns=["update_timestamp_utc"], errors="ignore")
        self.test_set = (
            self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
            .toPandas()
            .drop(columns=["update_timestamp_utc"], errors="ignore")
        )

        if self.client_id_col_name not in self.train_set.columns:
            logger.warning(
                f"Client ID column '{self.client_id_col_name}' not found in loaded train_set. "
                f"Available columns: {self.train_set.columns.tolist()}. "
                "High-risk adjustment in ModelWrapper might fail."
            )
        if self.client_id_col_name not in self.test_set.columns:
            logger.warning(
                f"Client ID column '{self.client_id_col_name}' not found in loaded test_set. "
                f"Available columns: {self.test_set.columns.tolist()}."
            )

        self.data_version = self.train_set_spark.version() if hasattr(self.train_set_spark, "version") else "0"

        # X_train/X_test for model training should only contain features
        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]

        # train_set_for_logging might be used for MLflow dataset logging, ensure it has features
        self.train_set_for_logging = self.train_set[self.num_features + self.cat_features]
        # self.test_set_for_logging = self.test_set[self.num_features + self.cat_features] # If needed

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
        additional_pip_deps = ["pyspark==3.5.0"]
        if self.code_paths:
            for package in self.code_paths:
                whl_name = package.split("/")[-1]
                additional_pip_deps.append(f"./code/{whl_name}")

        # Prepare artifacts dictionary for MLflow
        artifacts_to_log = {}
        if self.high_risk_artifact_path_for_logging:
            if not os.path.exists(self.high_risk_artifact_path_for_logging):
                logger.warning(
                    f"High-risk artifact path '{self.high_risk_artifact_path_for_logging}' does not exist. "
                    "It will not be logged with the model. ModelWrapper might fail at serving time."
                )
            else:
                artifacts_to_log[self.high_risk_artifact_name_in_mlflow] = self.high_risk_artifact_path_for_logging
        else:
            logger.warning(
                "high_risk_artifact_path_for_logging is not set. "
                "ModelWrapper might fail at serving time if it expects the artifact."
            )

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            # For evaluation, we use the raw model pipeline predictions before ModelWrapper adjustments
            y_pred_raw_model = self.pipeline.predict(self.X_test)

            # Evaluate metrics based on the raw model's performance
            accuracy = accuracy_score(self.y_test, y_pred_raw_model)
            precision = precision_score(self.y_test, y_pred_raw_model, zero_division=0)
            recall = recall_score(self.y_test, y_pred_raw_model, zero_division=0)
            f1 = f1_score(self.y_test, y_pred_raw_model, zero_division=0)

            logger.info(f"📊 Raw Model Accuracy: {accuracy}")
            logger.info(f"📊 Raw Model Precision: {precision}")
            logger.info(f"📊 Raw Model Recall: {recall}")
            logger.info(f"📊 Raw Model F1 Score: {f1}")

            mlflow.log_param("model_type", "LGBMClassifier with preprocessing and high-risk adjustment wrapper")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("raw_model_accuracy", accuracy)
            mlflow.log_metric("raw_model_precision", precision)
            mlflow.log_metric("raw_model_recall", recall)
            mlflow.log_metric("raw_model_f1score", f1)
            mlflow.log_param("client_id_col_name", self.client_id_col_name)
            mlflow.log_param("high_risk_artifact_name_in_mlflow", self.high_risk_artifact_name_in_mlflow)
            mlflow.log_param("client_id_col_in_artifact", self.client_id_col_in_artifact)
            mlflow.log_param("probability_adjustment_factor", self.probability_adjustment_factor)
            mlflow.log_param("probability_cap_value", self.probability_cap_value)

            # Log the dataset (features only for the core model training data)
            if dataset_type == "PandasDataset":
                dataset = mlflow.data.from_pandas(
                    self.train_set_for_logging,  # Already just features
                    name="train_set_features_default_detection",
                )
            elif dataset_type == "SparkDataset":
                train_set_spark_features = self.train_set_spark.select(self.num_features + self.cat_features)
                dataset = mlflow.data.from_spark(
                    train_set_spark_features,
                    name=f"{self.catalog_name}.{self.schema_name}.train_set_features_default_detection",
                    version=self.data_version,
                )
            else:
                raise ValueError("Unsupported dataset type.")
            mlflow.log_input(dataset, context="training")

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            # Define input_signature for ModelWrapper (includes client_id + features)
            # Assuming client ID is string, adjust if it's integer or other type.
            # The type should match the actual data type of self.client_id_col_name in self.train_set
            client_id_col_type = "string"  # Default assumption
            if self.client_id_col_name in self.train_set.columns:
                if pd.api.types.is_integer_dtype(self.train_set[self.client_id_col_name]):
                    client_id_col_type = "integer"
                elif pd.api.types.is_float_dtype(self.train_set[self.client_id_col_name]):
                    client_id_col_type = "double"

            input_schema_cols = [ColSpec(client_id_col_type, self.client_id_col_name)]

            int_num_features = ["X1", "X5", "X6", "X7", "X8", "X9", "X10", "X11"]
            float_num_features = ["X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23"]

            for feature in self.config.num_features:
                if feature in int_num_features:
                    input_schema_cols.append(ColSpec("integer", feature))
                elif feature in float_num_features:
                    input_schema_cols.append(ColSpec("double", feature))
                else:  # Default if not in specific lists, or raise error
                    input_schema_cols.append(ColSpec("double", feature))

            for feature in self.config.cat_features:
                input_schema_cols.append(ColSpec("string", feature))

            input_signature = Schema(input_schema_cols)

            # Define output_signature for ModelWrapper
            output_signature = Schema(
                [
                    ColSpec(client_id_col_type, self.client_id_col_name),
                    ColSpec("double", "adjusted_probability_default"),
                    ColSpec("integer", "final_prediction"),
                ]
            )
            signature = ModelSignature(inputs=input_signature, outputs=output_signature)

            # Prepare input example for ModelWrapper (client_id + features)
            if self.client_id_col_name not in self.train_set.columns:
                raise ValueError(
                    f"Client ID column '{self.client_id_col_name}' not found in self.train_set for input_example. "
                    f"Ensure it's loaded in load_data. Columns: {self.train_set.columns.tolist()}"
                )

            example_cols = [self.client_id_col_name] + self.num_features + self.cat_features
            # Ensure all example_cols are actually in self.train_set.columns before slicing
            missing_example_cols = [col for col in example_cols if col not in self.train_set.columns]
            if missing_example_cols:
                raise ValueError(
                    f"Missing columns for input_example: {missing_example_cols}. Available: {self.train_set.columns.tolist()}"
                )
            input_example_df = self.train_set[example_cols].iloc[0:1]

            # Instantiate ModelWrapper with necessary parameters
            wrapped_model = ModelWrapper(
                model=self.pipeline,  # The scikit-learn pipeline
                feature_cols=self.num_features + self.cat_features,
                client_id_col_name=self.client_id_col_name,
                high_risk_artifact_name_in_mlflow=self.high_risk_artifact_name_in_mlflow,
                client_id_col_in_artifact=self.client_id_col_in_artifact,
                probability_adjustment_factor=self.probability_adjustment_factor,
                probability_cap_value=self.probability_cap_value,
            )

            mlflow.pyfunc.log_model(
                python_model=wrapped_model,
                artifact_path="pyfunc-default-detection-model",
                artifacts=artifacts_to_log,  # Log the high-risk CSV
                code_paths=self.code_paths,
                conda_env=conda_env,
                signature=signature,
                input_example=input_example_df,
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
