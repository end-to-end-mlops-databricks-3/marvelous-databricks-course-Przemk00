# Databricks notebook source

"""Feature Lookup Model Training and Registration Pipeline module."""

import argparse
import os
import sys
from pathlib import Path

import mlflow
from loguru import logger
# from pyspark.dbutils import DBUtils # Not typically available in local scripts, consider removing if not essential
from pyspark.sql import SparkSession

sys.path.append(str(Path.cwd().parent / "src"))
from default_detection.config import ProjectConfig, Tags
from default_detection.models.feature_lookup_dd_model import FeatureLookupDDModel # Import the correct model

base_dir = os.path.abspath(str(Path.cwd().parent))
config_path = os.path.join(base_dir, "project_config.yml")
 
# Configure tracking uri
mlflow.set_tracking_uri("databricks://dbr-pg") # Assuming same tracking URI
mlflow.set_registry_uri("databricks-uc://dbr-pg") # Assuming same registry URI

try:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path", # This seems to be for config_path, can be simplified
        action="store",
        default=config_path, # Default to the determined config_path
        type=str,
        required=False, # Make it optional if we default it
    )
    parser.add_argument(
        "--env",
        action="store",
        default="dev", # Default environment
        type=str,
        required=False,
    )
    parser.add_argument(
        "--git_sha",
        action="store",
        default="0000000", # Placeholder default
        type=str,
        required=False,
    )
    parser.add_argument(
        "--job_run_id",
        action="store",
        default="local_run", # Placeholder default
        type=str,
        required=False,
    )
    parser.add_argument(
        "--branch",
        action="store",
        default="main", # Placeholder default
        type=str,
        required=False,
    )
    args = parser.parse_args()
except (argparse.ArgumentError, SystemExit):
    # Fallback for environments where parsing might fail (e.g. Databricks notebooks without %run)
    logger.warning("Argument parsing failed or not in CLI context, using default values.")
    args = argparse.Namespace(
        root_path=config_path, 
        env="dev", 
        git_sha="localgitsha", 
        job_run_id="localjobrun", 
        branch="localbranch"
    )

# COMMAND ----------

# Use args.root_path for config if provided, otherwise the default
effective_config_path = args.root_path
if not os.path.exists(effective_config_path):
    logger.warning(f"Config path {effective_config_path} from args not found, falling back to default relative path.")
    effective_config_path = "../project_config.yml" # Relative to script location

config = ProjectConfig.from_yaml(config_path=effective_config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id})

# COMMAND ----------

# Initialize FeatureLookupDDModel
logger.info("Initializing FeatureLookupDDModel...")

# Explicitly set MLFLOW_TRACKING_URI environment variable
# to ensure FeatureEngineeringClient picks it up.
# This should ideally match the profile used by mlflow.set_tracking_uri
# For databricks profile, the URI is 'databricks' or 'databricks://<profile_name>'
# We are using "databricks://dbr-pg" which implies the profile.
# The SDK might also need DATABRICKS_CONFIG_PROFILE="dbr-pg" if it doesn't infer from the URI.
# Let's ensure MLFLOW_TRACKING_URI is set to what MLflow expects for a profile.
# When using a profile, MLflow itself often just needs 'databricks' and DATABRICKS_CONFIG_PROFILE.
# However, since we set mlflow.set_tracking_uri("databricks://dbr-pg"),
# let's try setting MLFLOW_TRACKING_URI to "databricks" and ensure DATABRICKS_CONFIG_PROFILE is also set.

logger.info("Setting MLFLOW_TRACKING_URI and DATABRICKS_CONFIG_PROFILE environment variables.")
os.environ["MLFLOW_TRACKING_URI"] = "databricks" # Use 'databricks' for profile-based auth
os.environ["DATABRICKS_CONFIG_PROFILE"] = "dbr-pg" # Specify the profile

feature_lookup_model = FeatureLookupDDModel(
    config=config,
    tags=tags,
    spark=spark
    # code_paths are handled by fe.log_model for feature store models if needed,
    # or by the environment if running as part of a packaged application.
    # For this script, the model's own code is directly available via sys.path.
    # If the model logged by fe.log_model needs to bundle this script's dependencies,
    # that's a different concern handled at the fe.log_model step.
    # The primary concern for code_paths in the BaseDDModel was for mlflow.pyfunc.log_model.
    # Feature Engineering client handles dependencies differently.
)
logger.info("FeatureLookupDDModel initialized.")

# COMMAND ----------
logger.info("Creating feature table...")
feature_lookup_model.create_feature_table()
logger.info("Feature table creation process completed.")

# COMMAND ----------
logger.info("Loading data for FeatureLookupDDModel...")
feature_lookup_model.load_data()
logger.info("Data loading completed.")

# COMMAND ----------
logger.info("Performing feature engineering...")
feature_lookup_model.feature_engineering()
logger.info("Feature engineering completed.")

# COMMAND ----------
logger.info("Starting model training for FeatureLookupDDModel...")
feature_lookup_model.train() # This will use experiment_name_feature_lookup
logger.info("Model training completed.")

# COMMAND ----------
logger.info("Registering FeatureLookupDDModel...")
feature_lookup_model.register_model()
logger.info("FeatureLookupDDModel registration completed.")

# COMMAND ----------
logger.info("Feature Lookup Model pipeline finished successfully.")