"""Unit tests for PocessModeling."""

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
from default_detection.models.modeling_pipeline import DateFeatureEngineer, PocessModeling

mlflow.set_tracking_uri(TRACKING_URI)


def test_custom_model_init(config: ProjectConfig, tags: Tags, spark_session: SparkSession) -> None:
    """Test the initialization of PocessModeling..

    This function creates a PocessModeling instance and asserts that its attributes are of the correct types.
    :param config: Configuration for the project
    :param tags: Tags associated with the model
    :param spark_session: Spark session object
    """
    model = PocessModeling(config=config, tags=tags, spark=spark_session, code_paths=[])
    assert isinstance(model, PocessModeling)
    assert isinstance(model.config, ProjectConfig)
    assert isinstance(model.tags, dict)
    assert isinstance(model.spark, SparkSession)
    assert isinstance(model.code_paths, list)
    assert not model.code_paths


def test_prepare_features(mock_custom_model: PocessModeling) -> None:
    """Test that prepare_features method initializes pipeline components correctly..

    Verifies the preprocessor is a ColumnTransformer and pipeline contains expected
    ColumnTransformer and LGBMClassifier steps in sequence.
    :param mock_custom_model: Mocked PocessModeling instance for testing
    """
    mock_custom_model.prepare_features()

    assert isinstance(mock_custom_model.preprocessor, ColumnTransformer)
    assert isinstance(mock_custom_model.pipeline, Pipeline)
    assert isinstance(mock_custom_model.pipeline.steps, list)
    assert isinstance(mock_custom_model.pipeline.steps[0][1], DateFeatureEngineer)
    assert isinstance(mock_custom_model.pipeline.steps[1][1], ColumnTransformer)
    assert isinstance(mock_custom_model.pipeline.steps[2][1], LGBMClassifier)


def test_train(mock_custom_model: PocessModeling) -> None:
    """Test that train method configures pipeline with correct feature handling..

    Validates feature count matches configuration and feature names align with
    numerical/categorical features defined in model config.
    :param mock_custom_model: Mocked PocessModeling instance for testing
    """
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    expected_feature_names = [
        "cat__type_of_meal_plan_Meal Plan 1",
        "cat__type_of_meal_plan_Meal Plan 2",
        "cat__type_of_meal_plan_Meal Plan 3",
        "cat__type_of_meal_plan_Not Selected",
        "cat__required_car_parking_space_0",
        "cat__required_car_parking_space_1",
        "cat__room_type_reserved_Room_Type 1",
        "cat__room_type_reserved_Room_Type 2",
        "cat__room_type_reserved_Room_Type 3",
        "cat__room_type_reserved_Room_Type 4",
        "cat__room_type_reserved_Room_Type 5",
        "cat__room_type_reserved_Room_Type 6",
        "cat__room_type_reserved_Room_Type 7",
        "cat__market_segment_type_Aviation",
        "cat__market_segment_type_Complementary",
        "cat__market_segment_type_Corporate",
        "cat__market_segment_type_Offline",
        "cat__market_segment_type_Online",
        "cat__country_PL",
        "cat__country_UK",
        "remainder__no_of_adults",
        "remainder__no_of_children",
        "remainder__no_of_weekend_nights",
        "remainder__no_of_week_nights",
        "remainder__lead_time",
        "remainder__repeated_guest",
        "remainder__no_of_previous_cancellations",
        "remainder__no_of_previous_bookings_not_canceled",
        "remainder__avg_price_per_room",
        "remainder__no_of_special_requests",
        "remainder__month_sin",
        "remainder__month_cos",
        "remainder__is_first_quarter",
        "remainder__is_second_quarter",
        "remainder__is_third_quarter",
        "remainder__is_fourth_quarter",
    ]
    preprocessor = mock_custom_model.pipeline.named_steps["preprocessor"]

    assert len(list(preprocessor.get_feature_names_out())) == len(expected_feature_names)
    assert sorted(expected_feature_names) == sorted(preprocessor.get_feature_names_out())


def test_log_model_with_PandasDataset(mock_custom_model: PocessModeling) -> None:
    """Test model logging with PandasDataset validation..

    Verifies that the model's pipeline captures correct feature dimensions and names,
    then checks proper dataset type handling during model logging.
    :param mock_custom_model: Mocked PocessModeling instance for testing
    """
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    expected_feature_names = [
        "cat__type_of_meal_plan_Meal Plan 1",
        "cat__type_of_meal_plan_Meal Plan 2",
        "cat__type_of_meal_plan_Meal Plan 3",
        "cat__type_of_meal_plan_Not Selected",
        "cat__required_car_parking_space_0",
        "cat__required_car_parking_space_1",
        "cat__room_type_reserved_Room_Type 1",
        "cat__room_type_reserved_Room_Type 2",
        "cat__room_type_reserved_Room_Type 3",
        "cat__room_type_reserved_Room_Type 4",
        "cat__room_type_reserved_Room_Type 5",
        "cat__room_type_reserved_Room_Type 6",
        "cat__room_type_reserved_Room_Type 7",
        "cat__market_segment_type_Aviation",
        "cat__market_segment_type_Complementary",
        "cat__market_segment_type_Corporate",
        "cat__market_segment_type_Offline",
        "cat__market_segment_type_Online",
        "cat__country_PL",
        "cat__country_UK",
        "remainder__no_of_adults",
        "remainder__no_of_children",
        "remainder__no_of_weekend_nights",
        "remainder__no_of_week_nights",
        "remainder__lead_time",
        "remainder__repeated_guest",
        "remainder__no_of_previous_cancellations",
        "remainder__no_of_previous_bookings_not_canceled",
        "remainder__avg_price_per_room",
        "remainder__no_of_special_requests",
        "remainder__month_sin",
        "remainder__month_cos",
        "remainder__is_first_quarter",
        "remainder__is_second_quarter",
        "remainder__is_third_quarter",
        "remainder__is_fourth_quarter",
    ]
    preprocessor = mock_custom_model.pipeline.named_steps["preprocessor"]

    assert len(list(preprocessor.get_feature_names_out())) == len(expected_feature_names)
    assert sorted(expected_feature_names) == sorted(preprocessor.get_feature_names_out())

    mock_custom_model.log_model(dataset_type="PandasDataset")

    # Split the following part
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(mock_custom_model.experiment_name)
    assert experiment.name == mock_custom_model.experiment_name

    experiment_id = experiment.experiment_id
    assert experiment_id

    runs = client.search_runs(experiment_id, order_by=["start_time desc"], max_results=1)
    assert len(runs) == 1
    latest_run = runs[0]

    model_uri = f"runs:/{latest_run.info.run_id}/model"
    logger.info(f"{model_uri= }")

    assert model_uri


def test_register_model(mock_custom_model: PocessModeling) -> None:
    """Test the registration of a custom MLflow model..

    This function performs several operations on the mock custom model, including loading data,
    preparing features, training, and logging the model. It then registers the model and verifies
    its existence in the MLflow model registry.
    :param mock_custom_model: A mocked instance of the PocessModeling class.
    """
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    mock_custom_model.log_model(dataset_type="PandasDataset")

    mock_custom_model.register_model()

    client = MlflowClient()
    model_name = f"{mock_custom_model.catalog_name}.{mock_custom_model.schema_name}.dd_model"

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
    alias, version = model.aliases.popitem()
    assert alias == "latest-model"


def test_retrieve_current_run_metadata(mock_custom_model: PocessModeling) -> None:
    """Test retrieving the current run metadata from a mock custom model..

    This function verifies that the `retrieve_current_run_metadata` method
    of the `PocessModeling` class returns metrics and parameters as dictionaries.
    :param mock_custom_model: A mocked instance of the PocessModeling class.
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


def test_load_latest_model_and_predict(mock_custom_model: PocessModeling) -> None:
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
        "type_of_meal_plan",
        "required_car_parking_space",
        "room_type_reserved",
        "market_segment_type",
        "country",
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "lead_time",
        "repeated_guest",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "avg_price_per_room",
        "no_of_special_requests",
        "arrival_month",
        "Booking_ID",
        "Client_ID",
    ]
    data = [
        ["Meal Plan 1", 0, "Room_Type 1", "Online", "PL", 2, 1, 2, 1, 26, 0, 0, 0, 161, 0, 10, "INN25630", "ABCDE"],
        [
            "Meal Plan 1",
            0,
            "Room_Type 1",
            "Online",
            "PL",
            2,
            1,
            2,
            1,
            26,
            0,
            0,
            0,
            161,
            0,
            10,
            "INN25630",
            "client_banned",
        ],
    ]

    input_data = pd.DataFrame(data, columns=columns)

    cols_types = {
        "required_car_parking_space": "int32",
        "no_of_adults": "int32",
        "no_of_children": "int32",
        "no_of_weekend_nights": "int32",
        "no_of_week_nights": "int32",
        "lead_time": "int32",
        "repeated_guest": "int32",
        "no_of_previous_cancellations": "int32",
        "no_of_previous_bookings_not_canceled": "int32",
        "avg_price_per_room": "float32",
        "no_of_special_requests": "int32",
        "arrival_month": "int32",
    }

    input_data = input_data.astype(cols_types)

    for i in range(len(input_data)):
        row_df = input_data.iloc[[i]]
        predictions = mock_custom_model.load_latest_model_and_predict(input_data=row_df)

        client_id = row_df["Client_ID"].iloc[0]
        comment = predictions["Comment"].iloc[0]
        proba = predictions["Proba"].iloc[0]
        if client_id == "client_banned":
            assert comment == "Banned"
            assert proba == 1
        else:
            assert comment == "None"
        assert len(predictions.columns) == 3
        assert "Client_ID" in predictions.columns
        assert "Proba" in predictions.columns
        assert "Comment" in predictions.columns
