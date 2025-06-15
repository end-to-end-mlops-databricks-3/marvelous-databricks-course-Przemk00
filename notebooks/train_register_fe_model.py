# Databricks notebook source

# MAGIC %md
# MAGIC # Default Detection Model Training and Registration Pipeline (No Online Feature Store Lookups)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup and Imports

# COMMAND ----------
# COMMAND ----------
# MAGIC %pip install ../dist/default_detection-0.0.1-py3-none-any.whl --upgrade

# COMMAND ----------
import os
import sys
from pathlib import Path
import pandas as pd

import mlflow
from loguru import logger
from pyspark.sql import SparkSession

# Add src directory to sys.path to import custom modules
sys.path.append(str(Path.cwd().parent / "src"))

from default_detection.config import ProjectConfig, Tags
from default_detection.models.feature_lookup_model import DefaultDetectionModel

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------
# Configure MLflow tracking and registry URIs for Databricks environment
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Notebook parameters
ENV = "dev"
GIT_SHA = "notebook_run_sha_no_fs"
JOB_RUN_ID = "notebook_job_run_no_fs"
BRANCH = "notebook_branch_no_fs"

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
# MAGIC ## Initialize DefaultDetectionModel

# COMMAND ----------
logger.info("Initializing DefaultDetectionModel...")
try:
    model_instance = DefaultDetectionModel(
        config=config,
        tags=tags,
        spark=spark
    )
    logger.info("DefaultDetectionModel initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing DefaultDetectionModel: {e}")
    raise

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Data
# MAGIC The `load_data` method in `DefaultDetectionModel` loads all features directly.

# COMMAND ----------
logger.info("Loading data for DefaultDetectionModel...")
try:
    model_instance.load_data()
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
    model_instance.feature_engineering()
    logger.info("Feature engineering completed.")
except Exception as e:
    logger.error(f"Error during feature engineering: {e}")
    raise

# COMMAND ----------
# MAGIC %md
# MAGIC ## Train Model

# COMMAND ----------
logger.info("Starting model training for DefaultDetectionModel...")
try:
    model_instance.train() 
    logger.info("Model training completed.")
except Exception as e:
    logger.error(f"Error during model training: {e}")
    raise

# COMMAND ----------
# MAGIC %md
# MAGIC ## Register Model

# COMMAND ----------
logger.info("Registering DefaultDetectionModel...")
try:
    registered_version = model_instance.register_model()
    logger.info(f"DefaultDetectionModel registration completed. Version: {registered_version}")
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
    sample_spark_df = (
        spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
        .limit(10)
    )
    sample_pandas_df = sample_spark_df.toPandas()

    # Basic cleaning for the sample data
    for col in config.num_features:
        if col in sample_pandas_df.columns:
            sample_pandas_df[col] = pd.to_numeric(sample_pandas_df[col], errors='coerce').fillna(0)
        else: # Add missing numeric features with default 0
            logger.warning(f"Numeric feature '{col}' missing in sample, adding with 0.")
            sample_pandas_df[col] = 0
            
    for col in config.cat_features:
        if col in sample_pandas_df.columns:
            sample_pandas_df[col] = sample_pandas_df[col].fillna("missing").astype(str)
        else: # Add missing categorical features with default 'missing'
            logger.warning(f"Categorical feature '{col}' missing in sample, adding with 'missing'.")
            sample_pandas_df[col] = "missing"
            
    features_for_prediction = config.num_features + config.cat_features
    sample_pandas_df_for_input = sample_pandas_df[features_for_prediction]

    logger.info("Sample data for prediction (Pandas DataFrame head):")
    print(sample_pandas_df_for_input.head())
    logger.info(f"Sample data columns for input: {sample_pandas_df_for_input.columns.tolist()}")

    logger.info("Loading the latest registered model and making predictions...")
    prediction_model_loader = DefaultDetectionModel(config=config, tags=tags, spark=spark)
    
    predictions_series = prediction_model_loader.load_latest_model_and_predict(sample_pandas_df_for_input)
    
    logger.info("Predictions completed. Displaying results (Pandas Series):")
    print(predictions_series)

except Exception as e:
    logger.error(f"Error during sample prediction: {e}")

# COMMAND ----------
logger.info("Default Detection Model pipeline notebook finished successfully.")