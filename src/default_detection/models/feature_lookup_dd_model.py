"""FeatureLookUp model implementation for default detection."""

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from default_detection.config import ProjectConfig, Tags


class FeatureLookupDDModel:  # Renamed from DefaultDetectionFeatureLookupModel
    """A class to manage FeatureLookupModel for default detection."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        # Explicitly use the 'dbr-pg' profile, consistent with MLflow setup in the script
        # Ideally, this profile name would come from config or be passed to __init__
        _profile_name = "dbr-pg"
        _profile_name = "dbr-pg"
        logger.info(f"Initializing WorkspaceClient with profile: '{_profile_name}'")
        # Initialize WorkspaceClient first with the explicit profile
        self.workspace = WorkspaceClient(profile=_profile_name)
        logger.info("Initializing FeatureEngineeringClient")
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names
        self.feature_table_name = (
            f"{self.catalog_name}.{self.schema_name}.feature_lookup_dd_features"  # Updated table name
        )
        # self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_some_feature" # Example if needed

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_feature_lookup  # Using new config field
        self.tags = tags.dict()
        self.run_id = None  # Initialize run_id

    def create_feature_table(self) -> None:
        """Create or update the feature_lookup_dd_features table and populate it.

        This table stores features related to customer defaults for feature lookup.
        Assumes an 'Id' column exists in source tables for lookup.
        """
        num_feature_cols_sql = ", ".join([f"{col_name} DOUBLE" for col_name in self.num_features])
        cat_feature_cols_sql = ", ".join([f"{col_name} STRING" for col_name in self.cat_features])

        all_feature_cols_sql = ""
        if num_feature_cols_sql and cat_feature_cols_sql:
            all_feature_cols_sql = f", {num_feature_cols_sql}, {cat_feature_cols_sql}"
        elif num_feature_cols_sql:
            all_feature_cols_sql = f", {num_feature_cols_sql}"
        elif cat_feature_cols_sql:
            all_feature_cols_sql = f", {cat_feature_cols_sql}"

        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Id STRING NOT NULL{all_feature_cols_sql});
        """)
        self.spark.sql(
            f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT feature_lookup_dd_pk PRIMARY KEY(Id);"
        )  # Updated constraint name
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        feature_columns_to_select = ["Id"] + self.num_features + self.cat_features
        select_cols_str = ", ".join(feature_columns_to_select)

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT {select_cols_str} FROM {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT {select_cols_str} FROM {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info(f"✅ Feature table '{self.feature_table_name}' created and populated.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Drops columns that are now part of the feature table and will be looked up.
        Assumes 'Id' column is present for joining/lookup.
        """
        columns_in_feature_store = self.num_features + self.cat_features

        self.train_set_source = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(
            *columns_in_feature_store
        )
        self.test_set_source = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        self.train_set_source = self.train_set_source.withColumn("Id", self.train_set_source["Id"].cast("string"))
        if "Id" in self.test_set_source.columns:
            self.test_set_source["Id"] = self.test_set_source["Id"].astype(str)

        logger.info("✅ Data for feature lookup default detection successfully loaded.")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        Creates a training set using FeatureLookup.
        """
        features_to_lookup = self.num_features + self.cat_features

        self.training_set = self.fe.create_training_set(
            df=self.train_set_source,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=features_to_lookup,
                    lookup_key="Id",
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()

        self.X_train = self.training_df[features_to_lookup]
        self.y_train = self.training_df[self.target]

        self.X_test = self.test_set_source[features_to_lookup]
        self.y_test = self.test_set_source[self.target]

        logger.info("✅ Feature engineering for feature lookup default detection completed.")

    def train(self) -> None:
        """Train the classification model and log results to MLflow.

        Uses a pipeline with preprocessing and LightGBM classifier.
        """
        logger.info("🚀 Starting training for feature lookup default detection model...")

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_features)],
            remainder="passthrough",
        )

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**self.parameters))])

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)
            y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]

            acc = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            f1 = f1_score(self.y_test, y_pred)

            logger.info(f"📊 Accuracy: {acc}")
            logger.info(f"📊 ROC AUC: {roc_auc}")
            logger.info(f"📊 F1 Score: {f1}")

            mlflow.log_param("model_type", "LGBMClassifier_FeatureLookupDDModel")  # Updated model_type
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("f1_score", f1)

            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lgbm-pipeline-feature-lookup-dd",  # Updated artifact path
                training_set=self.training_set,
                signature=signature,
            )
        logger.info(f"✅ Training complete for FeatureLookupDDModel. Run ID: {self.run_id}")

    def register_model(self) -> str:
        """Register the trained model to MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        if not self.run_id:
            logger.error("❌ Run ID not set. Train the model first.")
            raise ValueError("Run ID is not set. Please train the model before registering.")

        model_name = f"{self.catalog_name}.{self.schema_name}.feature_lookup_dd_model"  # Updated model name
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lgbm-pipeline-feature-lookup-dd",  # Updated artifact path
            name=model_name,
            tags=self.tags,
        )

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias="FeatureLookupDDLatest",  # Updated alias
            version=latest_version,
        )
        logger.info(f"✅ Model registered: {model_name} version {latest_version} with alias 'FeatureLookupDDLatest'.")
        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'FeatureLookupDDLatest' and scores the batch.
        :param X: DataFrame containing the input features (including the lookup key 'Id').
        :return: DataFrame containing the predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.feature_lookup_dd_model@FeatureLookupDDLatest"  # Updated model name and alias

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        logger.info(f"✅ Predictions made using model: {model_uri}")
        return predictions
