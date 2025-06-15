"""Simplified model implementation for default detection, without feature store lookups."""

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# from sklearn.model_selection import train_test_split # Not currently used
from default_detection.config import ProjectConfig, Tags


class DefaultDetectionModel:
    """Manages a classification model for default detection.

    This version expects all features to be provided at training and inference time,
    without relying on Feature Store online lookups.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()  # Used by MlflowClient for context

        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.all_features = self.num_features + self.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters

        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Determine experiment name
        self.experiment_name = (
            getattr(self.config, "experiment_name_default_detection_model", None)
            or getattr(self.config, "experiment_name_base_model", None)
            or f"/Shared/{self.schema_name}/default_detection_model_experiment"
        )

        self.tags = tags.dict()
        self.run_id = None
        self.model_pipeline = None

        self.X_train: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.X_test: pd.DataFrame = None
        self.y_test: pd.Series = None
        logger.info(f"DefaultDetectionModel initialized. Experiment name: {self.experiment_name}")

    def load_data(self) -> None:
        """Load training and testing data directly.

        Assumes 'train_set' and 'test_set' tables in Delta Lake contain all necessary features and the target.
        """
        logger.info("Loading data for DefaultDetectionModel...")
        train_df_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        test_df_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")

        # Convert to Pandas
        train_df_pd = train_df_spark.toPandas()
        test_df_pd = test_df_spark.toPandas()
        logger.info(f"Loaded {len(train_df_pd)} training records and {len(test_df_pd)} test records.")

        # Basic Cleaning
        for col in self.num_features:
            train_df_pd[col] = pd.to_numeric(train_df_pd[col], errors="coerce").fillna(0)
            test_df_pd[col] = pd.to_numeric(test_df_pd[col], errors="coerce").fillna(0)

        for col in self.cat_features:
            train_df_pd[col] = train_df_pd[col].fillna("missing").astype(str)
            test_df_pd[col] = test_df_pd[col].fillna("missing").astype(str)

        train_df_pd[self.target] = pd.to_numeric(train_df_pd[self.target], errors="coerce").fillna(0).astype(int)
        test_df_pd[self.target] = pd.to_numeric(test_df_pd[self.target], errors="coerce").fillna(0).astype(int)

        self.X_train = train_df_pd[self.all_features]
        self.y_train = train_df_pd[self.target]
        self.X_test = test_df_pd[self.all_features]
        self.y_test = test_df_pd[self.target]

        logger.info("✅ Data loading and basic cleaning completed.")
        logger.info(f"X_train shape: {self.X_train.shape}, X_test shape: {self.X_test.shape}")

    def feature_engineering(self) -> None:
        """Perform feature engineering directly on the loaded Pandas DataFrames.

        The preprocessor in the `train` method handles scaling and encoding.
        Additional custom transformations can be added here if needed.
        """
        logger.info("✅ Feature engineering step (relies on preprocessor in train method).")

    def train(self) -> None:
        """Train the classification model and log results to MLflow."""
        if self.X_train is None or self.y_train is None:
            logger.error("Training data not loaded. Call load_data() and feature_engineering() first.")
            raise ValueError("Training data not available.")

        logger.info(f"Starting training for DefaultDetectionModel using experiment: {self.experiment_name}")

        # Ensure all feature columns exist in X_train before setting up ColumnTransformer
        for col_list_name, col_list in [("numeric", self.num_features), ("categorical", self.cat_features)]:
            missing_cols = [col for col in col_list if col not in self.X_train.columns]
            if missing_cols:
                msg = f"Missing {col_list_name} columns in X_train: {missing_cols}"
                logger.error(msg)
                raise ValueError(msg)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_features),
            ],
            remainder="drop",
        )

        self.model_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**self.parameters))]
        )

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            logger.info(f"Fitting model pipeline. X_train columns: {self.X_train.columns.tolist()}")
            self.model_pipeline.fit(self.X_train, self.y_train)
            logger.info("Pipeline fitting complete.")

            y_pred = self.model_pipeline.predict(self.X_test)
            y_pred_proba = self.model_pipeline.predict_proba(self.X_test)[:, 1]
            logger.info("Predictions on X_test complete.")

            acc = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            f1 = f1_score(self.y_test, y_pred)

            logger.info(f"📊 Accuracy: {acc:.4f}")
            logger.info(f"📊 ROC AUC: {roc_auc:.4f}")
            logger.info(f"📊 F1 Score: {f1:.4f}")

            mlflow.log_param("model_type", "LGBMClassifier_DefaultDetectionModel")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("f1_score", f1)

            signature = infer_signature(self.X_train, self.model_pipeline.predict(self.X_train))
            mlflow.sklearn.log_model(
                sk_model=self.model_pipeline,
                artifact_path="sklearn-model",
                signature=signature,
                input_example=self.X_train.head(5),
            )
            logger.info(f"✅ Training complete for DefaultDetectionModel. Run ID: {self.run_id}")

    def register_model(self) -> str:
        """Register the trained model to MLflow registry."""
        if not self.run_id:
            logger.error("Run ID not set. Train the model first.")
            raise ValueError("Run ID is not set. Please train the model before registering.")

        model_name = f"{self.catalog_name}.{self.schema_name}.default_detection_model"
        model_uri = f"runs:/{self.run_id}/sklearn-model"

        logger.info(f"Registering model '{model_name}' from URI '{model_uri}'")
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=self.tags,
        )
        latest_version = registered_model.version

        client = MlflowClient()
        alias_name = "DefaultDetectionLatest"
        client.set_registered_model_alias(
            name=model_name,
            alias=alias_name,
            version=latest_version,
        )
        logger.info(f"✅ Model registered: {model_name} version {latest_version} with alias '{alias_name}'.")
        return latest_version

    def load_latest_model_and_predict(self, X: pd.DataFrame) -> pd.Series:  # Expects Pandas DataFrame
        """Load the trained model from MLflow and make predictions. Expects a Pandas DataFrame."""
        if not isinstance(X, pd.DataFrame):
            msg = "Input X to predict must be a Pandas DataFrame."
            logger.error(msg)
            raise TypeError(msg)

        X_df = X.copy()  # Work on a copy

        # Ensure all required feature columns are present and in the correct order
        missing_cols = [col for col in self.all_features if col not in X_df.columns]
        if missing_cols:
            msg = f"Input data for prediction is missing columns: {missing_cols}"
            logger.error(msg)
            raise ValueError(msg)

        X_df = X_df[self.all_features]  # Ensure column order

        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.default_detection_model@DefaultDetectionLatest"
        logger.info(f"Loading model from URI: {model_uri}")
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        logger.info("Making predictions with loaded model...")
        predictions = loaded_model.predict(X_df)
        logger.info(f"✅ Predictions made using model: {model_uri}")
        return pd.Series(predictions, name="predictions")
