"""FeatureLookUp model implementation for default detection."""

import mlflow
import pandas as pd
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from default_detection.config import ProjectConfig, Tags


class FeatureLookUpModel:
    """A class to manage FeatureLookUpModel for default detection."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.feature_lookup_features"
        self.experiment_name = self.config.experiment_name_feature_lookup
        self.tags = tags.dict()
        self.run_id = None

    def create_feature_table(self) -> None:
        """Create or update the feature_lookup_features table and populate it with cleaned data."""
        # 1. Define the table schema and create the table with a primary key.
        num_feature_cols_sql = ", ".join([f"`{col_name}` DOUBLE" for col_name in self.num_features])
        cat_feature_cols_sql = ", ".join([f"`{col_name}` STRING" for col_name in self.cat_features])
        all_feature_cols_sql = f", {num_feature_cols_sql}, {cat_feature_cols_sql}"

        self.spark.sql(f"CREATE OR REPLACE TABLE {self.feature_table_name} (Id STRING NOT NULL{all_feature_cols_sql});")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT feature_lookup_pk PRIMARY KEY(Id);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        # 2. Load the raw data from source tables.
        logger.info("Loading raw data for feature table creation...")
        train_df_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        test_df_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        all_data_spark = train_df_spark.unionByName(test_df_spark).distinct()

        # 3. Clean the data using Spark functions.
        logger.info("Cleaning data before inserting into feature table...")
        cleaned_df = all_data_spark
        for c in self.num_features:
            cleaned_df = cleaned_df.withColumn(c, when(col(c).isNull(), 0).otherwise(col(c).cast("double")))
        for c in self.cat_features:
            cleaned_df = cleaned_df.withColumn(c, when(col(c).isNull(), "missing").otherwise(col(c).cast("string")))

        # 4. Insert the cleaned data into the feature table without dropping it.
        feature_columns_to_select = ["Id"] + self.num_features + self.cat_features
        cleaned_df.select(feature_columns_to_select).createOrReplaceTempView("cleaned_features_for_insert")

        self.spark.sql(f"INSERT OVERWRITE {self.feature_table_name} SELECT * FROM cleaned_features_for_insert")

        logger.info(f"✅ Feature table '{self.feature_table_name}' created and populated with cleaned data.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables."""
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
        """Perform feature engineering by linking data with feature tables."""
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

        # Data from the feature store is now clean.
        self.X_train = self.training_df[features_to_lookup]
        self.y_train = self.training_df[self.target].astype(int)

        # The test set still needs to be cleaned as it's loaded directly.
        self.X_test = self.test_set_source[features_to_lookup].copy()
        self.y_test = self.test_set_source[self.target].copy()

        for c in self.num_features:
            self.X_test[c] = pd.to_numeric(self.X_test[c], errors="coerce").fillna(0)
        for c in self.cat_features:
            self.X_test[c] = self.X_test[c].fillna("missing").astype(str)
        self.y_test = pd.to_numeric(self.y_test, errors="coerce").fillna(0).astype(int)

        logger.info("✅ Feature engineering completed.")

    def train(self) -> None:
        """Train the classification model and log results to MLflow."""
        logger.info("🚀 Starting training for feature lookup default detection model...")
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", self.num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_features),
            ],
            remainder="drop",
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LGBMClassifier(**self.parameters))])
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            logger.info("Pipeline fitting complete.")
            y_pred = pipeline.predict(self.X_test)
            y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]
            logger.info("Predictions on X_test complete.")
            acc = accuracy_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            f1 = f1_score(self.y_test, y_pred)
            logger.info(f"📊 Accuracy: {acc:.4f}")
            logger.info(f"📊 ROC AUC: {roc_auc:.4f}")
            logger.info(f"📊 F1 Score: {f1:.4f}")
            mlflow.log_param("model_type", "LGBMClassifier_FeatureLookUpModel")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("f1_score", f1)
            signature = infer_signature(self.X_train, pipeline.predict(self.X_train))
            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lgbm-pipeline-feature-lookup",
                training_set=self.training_set,
                signature=signature,
            )
        logger.info(f"✅ Training complete for FeatureLookUpModel. Run ID: {self.run_id}")

    def register_model(self) -> str:
        """Register the trained model to MLflow registry."""
        if not self.run_id:
            logger.error("❌ Run ID not set. Train the model first.")
            raise ValueError("Run ID is not set. Please train the model before registering.")
        model_name = f"{self.catalog_name}.{self.schema_name}.feature_lookup_model"
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lgbm-pipeline-feature-lookup",
            name=model_name,
            tags=self.tags,
        )
        latest_version = registered_model.version
        client = MlflowClient()
        alias_name = "FeatureLookUpLatest"
        client.set_registered_model_alias(
            name=model_name,
            alias=alias_name,
            version=latest_version,
        )
        logger.info(f"✅ Model registered: {model_name} version {latest_version} with alias '{alias_name}'.")
        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow and make predictions."""
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.feature_lookup_model@FeatureLookUpLatest"
        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        logger.info(f"✅ Predictions made using model: {model_uri}")
        return predictions
