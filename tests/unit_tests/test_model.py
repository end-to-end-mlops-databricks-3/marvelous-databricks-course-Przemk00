"""Unit tests for DefaultDetectionModeling."""

import mlflow
import pandas as pd
from conftest import TRACKING_URI
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from default_detection.config import ProjectConfig, Tags
from default_detection.models.default_detection_pipeline import DefaultDetectionModeling

mlflow.set_tracking_uri(TRACKING_URI)


def test_custom_model_init(config: ProjectConfig, tags: Tags, spark_session: SparkSession) -> None:
    """Test the initialization of DefaultDetectionModeling..

    This function creates a DefaultDetectionModeling instance and asserts that its attributes are of the correct types.
    :param config: Configuration for the project
    :param tags: Tags associated with the model
    :param spark_session: Spark session object
    """
    model = DefaultDetectionModeling(config=config, tags=tags, spark=spark_session, code_paths=[])
    assert isinstance(model, DefaultDetectionModeling)
    assert isinstance(model.config, ProjectConfig)
    assert isinstance(model.tags, dict)
    assert isinstance(model.spark, SparkSession)
    assert isinstance(model.code_paths, list)
    assert not model.code_paths


def test_prepare_features(mock_custom_model: DefaultDetectionModeling) -> None:
    """Test that prepare_features method initializes pipeline components correctly..

    Verifies the preprocessor is a ColumnTransformer and pipeline contains expected
    ColumnTransformer and LGBMClassifier steps in sequence.
    :param mock_custom_model: Mocked DefaultDetectionModeling instance for testing
    """
    mock_custom_model.prepare_features()

    assert isinstance(mock_custom_model.preprocessor, ColumnTransformer)
    assert isinstance(mock_custom_model.pipeline, Pipeline)
    assert isinstance(mock_custom_model.pipeline.steps, list)
    # DateFeatureEngineer is removed for the default detection model
    assert isinstance(mock_custom_model.pipeline.steps[0][1], ColumnTransformer)  # preprocessor
    assert isinstance(mock_custom_model.pipeline.steps[1][1], LGBMClassifier)  # classifier


def test_train(mock_custom_model: DefaultDetectionModeling, expected_feature_names: list[str]) -> None:
    """Test that train method configures pipeline with correct feature handling..

    Validates feature count matches configuration and feature names align with
    numerical/categorical features defined in model config.
    :param mock_custom_model: Mocked DefaultDetectionModeling instance for testing
    :param expected_feature_names: Fixture providing the expected feature names after preprocessing
    """
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()

    preprocessor = mock_custom_model.pipeline.named_steps["preprocessor"]

    assert len(list(preprocessor.get_feature_names_out())) == len(expected_feature_names)
    assert sorted(expected_feature_names) == sorted(preprocessor.get_feature_names_out())


def test_log_model_with_PandasDataset(
    mock_custom_model: DefaultDetectionModeling, expected_feature_names: list[str]
) -> None:
    """Test model logging with PandasDataset validation..

    Verifies that the model's pipeline captures correct feature dimensions and names,
    then checks proper dataset type handling during model logging.
    :param mock_custom_model: Mocked DefaultDetectionModeling instance for testing
    :param expected_feature_names: Fixture providing the expected feature names after preprocessing
    """
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()

    preprocessor = mock_custom_model.pipeline.named_steps["preprocessor"]

    assert len(list(preprocessor.get_feature_names_out())) == len(expected_feature_names)
    assert sorted(expected_feature_names) == sorted(preprocessor.get_feature_names_out())

    mock_custom_model.log_model(dataset_type="PandasDataset")

    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(mock_custom_model.experiment_name)
    assert experiment.name == mock_custom_model.experiment_name

    experiment_id = experiment.experiment_id
    assert experiment_id

    runs = client.search_runs(experiment_id, order_by=["start_time desc"], max_results=1)
    assert len(runs) == 1
    latest_run = runs[0]

    model_uri = f"runs:/{latest_run.info.run_id}/pyfunc-default-detection-model"
    logger.info(f"{model_uri= }")

    assert model_uri
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    assert loaded_model is not None


def test_register_model(mock_custom_model: DefaultDetectionModeling) -> None:
    """Test the registration of a custom MLflow model..

    This function performs several operations on the mock custom model, including loading data,
    preparing features, training, and logging the model. It then registers the model and verifies
    its existence in the MLflow model registry.
    :param mock_custom_model: A mocked instance of the DefaultDetectionModeling class.
    """
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    mock_custom_model.log_model(dataset_type="PandasDataset")

    mock_custom_model.register_model()

    client = MlflowClient()
    model_name = f"{mock_custom_model.catalog_name}.{mock_custom_model.schema_name}.default_detection_model"

    try:
        model = client.get_registered_model(model_name)
        logger.info(f"Model '{model_name}' is registered.")
        logger.info(f"Latest version: {model.latest_versions[-1].version}")
        logger.info(f"{model.name = }")
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            logger.error(f"Model '{model_name}' is not registered.")
        else:
            raise e

    assert isinstance(model, RegisteredModel)
    # Updated alias to match the one set in modeling_pipeline.py
    # Ensure 'Challenger' alias exists and points to the latest version.
    assert "Challenger" in model.aliases
    assert int(model.aliases["Challenger"]) == model.latest_versions[-1].version


def test_retrieve_current_run_metadata(mock_custom_model: DefaultDetectionModeling) -> None:
    """Test retrieving the current run metadata from a mock custom model..

    This function verifies that the `retrieve_current_run_metadata` method
    of the `DefaultDetectionModeling` class returns metrics and parameters as dictionaries.
    :param mock_custom_model: A mocked instance of the DefaultDetectionModeling class.
    """
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    mock_custom_model.log_model(dataset_type="PandasDataset")

    metrics, params = mock_custom_model.retrieve_current_run_metadata()
    assert isinstance(metrics, dict)
    assert metrics
    assert isinstance(params, dict)
    assert params


def test_load_latest_model_and_predict(mock_custom_model: DefaultDetectionModeling) -> None:
    """Test the process of loading the latest model and making predictions..

    This function performs the following steps:
    - Loads data using the provided custom model.
    - Prepares features and trains the model.
    - Logs and registers the trained model.
    - Extracts input data from the test set and makes predictions using the latest model.
    :param mock_custom_model: Instance of a custom machine learning model with methods for data
                            loading, feature preparation, training, logging, and prediction.
    """
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    mock_custom_model.log_model(dataset_type="PandasDataset")
    mock_custom_model.register_model()

    columns = [
        "ID",  # Added ID column
        "X1",
        "X2",
        "X3",
        "X4",
        "X5",
        "X6",
        "X7",
        "X8",
        "X9",
        "X10",
        "X11",
        "X12",
        "X13",
        "X14",
        "X15",
        "X16",
        "X17",
        "X18",
        "X19",
        "X20",
        "X21",
        "X22",
        "X23",
    ]
    data = [
        # Sample row 1 (corresponds to ID, X1-X23)
        [
            1,  # Added ID value
            50000,
            "1",
            "2",
            "1",
            30,
            0,
            0,
            0,
            0,
            0,
            0,
            50000.0,
            50000.0,
            50000.0,
            50000.0,
            50000.0,
            50000.0,
            2000.0,
            2000.0,
            2000.0,
            2000.0,
            2000.0,
            2000.0,
        ],
        # Sample row 2
        [
            2,  # Added ID value
            20000,
            "2",
            "1",
            "2",
            25,
            1,
            2,
            0,
            0,
            0,
            0,
            1000.0,
            800.0,
            1200.0,
            1000.0,
            0.0,
            0.0,
            500.0,
            300.0,
            700.0,
            0.0,
            0.0,
            0.0,
        ],
    ]

    input_data = pd.DataFrame(data, columns=columns)

    cols_types = {
        "ID": "int32",  # Added ID type
        "X1": "int32",
        "X5": "int32",
        "X6": "int32",
        "X7": "int32",
        "X8": "int32",
        "X9": "int32",
        "X10": "int32",
        "X11": "int32",
        "X12": "float32",
        "X13": "float32",
        "X14": "float32",
        "X15": "float32",
        "X16": "float32",
        "X17": "float32",
        "X18": "float32",
        "X19": "float32",
        "X20": "float32",
        "X21": "float32",
        "X22": "float32",
        "X23": "float32",
        "X2": "str",
        "X3": "str",
        "X4": "str",
    }

    input_data = input_data.astype(cols_types)

    predictions = mock_custom_model.load_latest_model_and_predict(input_data=input_data)

    assert predictions is not None
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions.columns) == 3
    assert mock_custom_model.client_id_col_name in predictions.columns  # "ID"
    assert "adjusted_probability_default" in predictions.columns
    assert "final_prediction" in predictions.columns
    assert len(predictions) == len(input_data)
    # Check dtypes for the new columns
    assert (
        predictions[mock_custom_model.client_id_col_name].dtype
        == input_data[mock_custom_model.client_id_col_name].dtype
    )
    assert predictions["final_prediction"].dtype == "int64" or predictions["final_prediction"].dtype == "int32"
    assert predictions["adjusted_probability_default"].dtype == "float64"
