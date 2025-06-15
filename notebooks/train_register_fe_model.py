# Databricks notebook source

# MAGIC %md
# MAGIC # Feature Lookup Model Training and Registration Pipeline

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup and Imports

# COMMAND ----------
# COMMAND ----------
%pip install ../dist/default_detection-0.0.1-py3-none-any.whl

# COMMAND ----------
import os
import sys
from pathlib import Path

import mlflow
from loguru import logger
from pyspark.sql import SparkSession

# Add src directory to sys.path to import custom modules
sys.path.append(str(Path.cwd().parent / "src"))

from default_detection.config import ProjectConfig, Tags
from default_detection.models.feature_lookup_model import FeatureLookUpModel

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------
# Configure MLflow tracking and registry URIs for Databricks environment
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Notebook parameters
ENV = "dev"
GIT_SHA = "notebook_run_sha"
JOB_RUN_ID = "notebook_job_run"
BRANCH = "notebook_branch"

# Path to the project configuration file
config_file_path = "../project_config.yml"

logger.info(f"Using environment: {ENV}")
logger.info(f"Project config path: {config_file_path}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Initialize Spark, Load Config, and Define Tags

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

try:
    config = ProjectConfig.from_yaml(config_path=config_file_path, env=ENV)
    logger.info("Project configuration loaded successfully.")
except FileNotFoundError:
    logger.error(f"Configuration file not found at: {config_file_path}")
    raise
except Exception as e:
    logger.error(f"Error loading project configuration: {e}")
    raise

tags_data = {"git_sha": GIT_SHA, "branch": BRANCH, "job_run_id": JOB_RUN_ID}
tags = Tags(**tags_data)
logger.info(f"Tags for MLflow run: {tags.dict()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Initialize FeatureLookUpModel

# COMMAND ----------
logger.info("Initializing FeatureLookUpModel...")
try:
    feature_lookup_model = FeatureLookUpModel(
        config=config,
        tags=tags,
        spark=spark
    )
    logger.info("FeatureLookUpModel initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing FeatureLookUpModel: {e}")
    raise

# COMMAND ----------
# MAGIC %md
# MAGIC ## Create Feature Table

# COMMAND ----------
logger.info("Creating feature table...")
try:
    feature_lookup_model.create_feature_table()
    logger.info("Feature table creation process completed.")
except Exception as e:
    logger.error(f"Error creating feature table: {e}")
    raise

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------
logger.info("Loading data for FeatureLookUpModel...")
try:
    feature_lookup_model.load_data()
    logger.info("Data loading completed.")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise

# COMMAND ----------
# MAGIC %md
# MAGIC ## Perform Feature Engineering

# COMMAND ----------
logger.info("Performing feature engineering...")
try:
    feature_lookup_model.feature_engineering()
    logger.info("Feature engineering completed.")
except Exception as e:
    logger.error(f"Error during feature engineering: {e}")
    raise

# COMMAND ----------
# MAGIC %md
# MAGIC ## Train Model

# COMMAND ----------
logger.info("Starting model training for FeatureLookUpModel...")
try:
    feature_lookup_model.train() # This will use experiment_name_feature_lookup from config
    logger.info("Model training completed.")
except Exception as e:
    logger.error(f"Error during model training: {e}")
    raise

# COMMAND ----------
# MAGIC %md
# MAGIC ## Register Model

# COMMAND ----------
logger.info("Registering FeatureLookUpModel...")
try:
    registered_version = feature_lookup_model.register_model()
    logger.info(f"FeatureLookUpModel registration completed. Version: {registered_version}")
except Exception as e:
    logger.error(f"Error during model registration: {e}")
    raise

# COMMAND ----------
# MAGIC %md
# MAGIC ## Optional: Load Registered Model and Predict
# MAGIC This section demonstrates loading the newly registered model and making predictions.

# COMMAND ----------
logger.info("Loading sample data for prediction...")
try:
    # Columns to drop are those that will be looked up by the feature store, plus the target.
    features_to_drop_for_prediction = config.num_features + config.cat_features + [config.target]
    
    sample_test_df_for_prediction = (
        spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
        .drop(*features_to_drop_for_prediction)
        .withColumnRenamed("ID", "Id") # Rename ID to Id for feature lookup
        .limit(10) # Using a small sample for demonstration
    )
    
    if "Id" not in sample_test_df_for_prediction.columns:
        logger.warning("Prediction sample data does not contain 'Id' column. This might be needed for feature lookup.")

    logger.info("Sample data for prediction:")
    sample_test_df_for_prediction.show()

    logger.info("Loading the latest registered model and making predictions...")
    # Re-initialize a FeatureLookUpModel instance to load the registered model
    prediction_model_loader = FeatureLookUpModel(config=config, tags=tags, spark=spark)
    
    predictions_df = prediction_model_loader.load_latest_model_and_predict(sample_test_df_for_prediction)
    
    logger.info("Predictions completed. Displaying results:")
    predictions_df.show()

except Exception as e:
    logger.error(f"Error during sample prediction: {e}")
    # dbutils.notebook.exit(f"Error during sample prediction: {e}") # Optional: exit notebook on error

# COMMAND ----------
logger.info("Feature Lookup Model pipeline notebook finished successfully.")