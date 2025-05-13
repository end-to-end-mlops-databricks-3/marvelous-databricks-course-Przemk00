import argparse
import pandas as pd # Added for pd.read_excel
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from default_detection.config import ProjectConfig
from default_detection.data_processor import DataProcessor
# generate_synthetic_data was imported but not used in the original script from commit b71c763
# and is not present in the DataProcessor class we defined.
from marvelous.logging import setup_logging 
from marvelous.timer import Timer

config_path = "../project_config.yml" # Assumes script is in 'scripts/' and config is in project root

config = ProjectConfig.from_yaml(config_path=config_path, env="dev")

# Adjusted log file path to be local
log_file_path = "../logs/01.preprocess_data.log" 
setup_logging(log_file=log_file_path)
logger.info(f"Logging to: {log_file_path}")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))


# Initialize SparkSession (DataProcessor expects it, even if initial load is pandas)
spark = SparkSession.builder.appName("DefaultDetectionPreprocessing").getOrCreate()

# Load the dataset using pandas
data_file_path = "../data/data.xls" # Relative to the script's location
logger.info(f"Loading data from: {data_file_path}")
try:
    df = pd.read_excel(data_file_path)
except ImportError as e:
    logger.error(f"Failed to read Excel file '{data_file_path}'. Ensure 'openpyxl' or 'xlrd' is installed: {e}")
    raise
except FileNotFoundError:
    logger.error(f"Data file not found at: {data_file_path}")
    raise
logger.info(f"Successfully loaded data. Shape: {df.shape}")

# Preprocess the data
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

logger.info("Script 01.preprocess_data.py finished.")