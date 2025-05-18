# Databricks notebook source
import argparse
import pandas as pd
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from default_detection.config import ProjectConfig
from default_detection.data_processor import DataProcessor
from marvelous.logging import setup_logging 
from marvelous.timer import Timer

# COMMAND ----------
config_path = "../project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

log_file_path = "../logs/01.preprocess_data.log"
setup_logging(log_file=log_file_path)
logger.info(f"Logging to: {log_file_path}")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

spark = SparkSession.builder.appName("DefaultDetectionPreprocessing").getOrCreate()

data_file_path = "../data/data.csv"
logger.info(f"Loading data from: {data_file_path}")
try:
    df = pd.read_csv(data_file_path, sep=';', header=0, skiprows=[1])
except FileNotFoundError:
    logger.error(f"Data file not found at: {data_file_path}")
    raise
logger.info(f"Successfully loaded data. Shape: {df.shape}")

with Timer() as preprocess_timer:
    data_processor = DataProcessor(df, config, spark)
    data_processor.preprocess()

logger.info(f"Data preprocessing: {preprocess_timer}")

# Split the data
X_train, X_test = data_processor.split_data()
logger.info(f"Training set shape: {X_train.shape}")
logger.info(f"Test set shape: {X_test.shape}")

# Save to catalog
logger.info("Saving data to catalog")
data_processor.save_to_catalog(X_train, X_test)

logger.info("Script finished.")
# COMMAND ----------

