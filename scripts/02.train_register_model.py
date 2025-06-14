# Databricks notebook source

"""Modeling Pipeline module."""

import argparse
import os
import sys
from pathlib import Path

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

sys.path.append(str(Path.cwd().parent / "src"))
from default_detection.config import ProjectConfig, Tags
from default_detection.models.modeling_pipeline import PocessModeling

base_dir = os.path.abspath(str(Path.cwd().parent))
config_path = os.path.join(base_dir, "project_config.yml")
 
# Configure tracking uri
mlflow.set_tracking_uri("databricks://dbr-pg")
mlflow.set_registry_uri("databricks-uc://dbr-pg")

try:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        action="store",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--env",
        action="store",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--git_sha",
        action="store",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--job_run_id",
        action="store",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--branch",
        action="store",
        default=None,
        type=str,
        required=True,
    )
    args = parser.parse_args()
except (argparse.ArgumentError, SystemExit):
    args = argparse.Namespace(root_path=config_path, env="dev", git_sha="123", job_run_id="1234567890", branch="przemekg")


# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2","job_run_id": "1234567890"})
# COMMAND ----------

# Initialize model
modeling_ppl = PocessModeling(
    config=config, tags=tags, spark=spark, code_paths=["../src/default_detection/models/modeling_pipeline.py"]
)
logger.info("Model initialized.")

# COMMAND ----------
# Load data and prepare features
modeling_ppl.load_data()
modeling_ppl.prepare_features()
logger.info("Loaded data, prepared features.")

# Train + log the model (runs everything including MLflow logging)
if config.hyperparameters_tuning:
    modeling_ppl.tune_hyperparameters()
modeling_ppl.train()
modeling_ppl.log_model()
logger.info("Model training completed.")

modeling_ppl.register_model()
logger.info("Registered model")
# COMMAND ----------
