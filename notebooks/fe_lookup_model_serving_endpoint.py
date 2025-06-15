# Databricks notebook source
# MAGIC %pip install ../dist/default_detection-0.0.1-py3-none-any.whl databricks-sdk --upgrade

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import os
import time
import json
import requests

from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize
from databricks.sdk.errors import ResourceDoesNotExist, OperationFailed
import pandas as pd

from default_detection.config import ProjectConfig

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup Environment and Configuration

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Get Databricks token and host
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
config_env = "dev"
config_file_path = "../project_config.yml"
project_config = ProjectConfig.from_yaml(config_path=config_file_path, env=config_env)
print(f"Project configuration loaded for environment: {config_env}")

catalog_name = project_config.catalog_name
schema_name = project_config.schema_name
model_name_short = "default_detection_model" # Used for served model entity name
registered_model_name = f"{catalog_name}.{schema_name}.{model_name_short}"
model_alias = "DefaultDetectionLatest"
endpoint_name = "default-detection-model-serving"

print(f"Target Model: {registered_model_name}@{model_alias}")
print(f"Endpoint Name: {endpoint_name}")

w = WorkspaceClient()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Deploy or Update Model Serving Endpoint

# COMMAND ----------
def deploy_or_update_serving_endpoint(
    workspace_client: WorkspaceClient, 
    endpoint_name: str, 
    model_name: str, 
    model_alias: str,
    workload_size: str = "Small"
):
    """
    Deploys a new model serving endpoint or updates an existing one.
    Resolves model alias to a specific version number for endpoint configuration.
    """
    print(f"Resolving alias '{model_alias}' for model '{model_name}' to a specific version...")
    try:
        model_version_info = workspace_client.model_versions.get_by_alias(model_name, model_alias)
        actual_model_version = model_version_info.version
        print(f"Resolved alias '{model_alias}' to version '{actual_model_version}' for model '{model_name}'.")
    except Exception as e:
        dbutils.notebook.exit(f"Error resolving model alias '{model_alias}' for model '{model_name}': {e}")

    served_model_name_in_endpoint = f"{model_name_short.replace('_', '-')}-v{actual_model_version}"
    served_model_config = ServedModelInput(
        name=served_model_name_in_endpoint,
        model_name=model_name,
        model_version=actual_model_version,
        workload_size=ServedModelInputWorkloadSize[workload_size.upper()],
        scale_to_zero_enabled=True
    )
    
    try:
        existing_endpoint = workspace_client.serving_endpoints.get(name=endpoint_name)
        print(f"Endpoint '{endpoint_name}' already exists.")
        
        current_served_model = None
        if existing_endpoint.config and existing_endpoint.config.served_models:
            current_served_model = next((
                sm for sm in existing_endpoint.config.served_models
                if sm.model_name == model_name and sm.model_version == actual_model_version
            ), None)

        if current_served_model and \
           current_served_model.workload_size == ServedModelInputWorkloadSize[workload_size.upper()] and \
           existing_endpoint.state and existing_endpoint.state.ready == "READY":
            print(f"Endpoint '{endpoint_name}' is already serving model '{model_name}' v{actual_model_version} and is READY. No update needed.")
            return

        print(f"Updating endpoint '{endpoint_name}' to serve model '{model_name}' v{actual_model_version}.")
        workspace_client.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_models=[served_model_config]
        )
        print(f"Endpoint '{endpoint_name}' update initiated.")
        
    except ResourceDoesNotExist:
        print(f"Endpoint '{endpoint_name}' not found. Creating...")
        endpoint_core_config = EndpointCoreConfigInput(name=endpoint_name, served_models=[served_model_config])
        try:
            workspace_client.serving_endpoints.create_and_wait(
                name=endpoint_name,
                config=endpoint_core_config
            )
            print(f"Endpoint '{endpoint_name}' creation initiated.")
        except Exception as creation_exception:
            dbutils.notebook.exit(f"ERROR during endpoint CREATION for '{endpoint_name}': {creation_exception}")

    except Exception as e:
        dbutils.notebook.exit(f"ERROR interacting with endpoint '{endpoint_name}': {e}")

    # Confirm final state after create/update_and_wait
    try:
        final_status = workspace_client.serving_endpoints.get(name=endpoint_name)
        if final_status.state and (final_status.state.ready == "READY" or final_status.state.config_update == "READY"):
            print(f"Confirmed: Endpoint '{endpoint_name}' is ready or update completed.")
        else:
            print(f"Warning: Endpoint '{endpoint_name}' final state is {final_status.state}. Manual check advised.")
    except Exception as e_status_check:
        print(f"Warning: Could not perform final status check for endpoint '{endpoint_name}': {e_status_check}")

# Deploy or update the endpoint
deploy_or_update_serving_endpoint(
    workspace_client=w,
    endpoint_name=endpoint_name,
    model_name=registered_model_name,
    model_alias=model_alias,
    workload_size="Small" # Or other sizes like "Medium", "Large"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Prepare Sample Request Data

# COMMAND ----------
dataframe_records_payload = []
try:
    sample_spark_df = spark.table(f"{project_config.catalog_name}.{project_config.schema_name}.test_set").limit(5)
    sample_pandas_df = sample_spark_df.toPandas()

    all_expected_features = project_config.num_features + project_config.cat_features

    for col in all_expected_features:
        if col not in sample_pandas_df.columns:
            # Add missing feature columns with default values
            if col in project_config.num_features:
                sample_pandas_df[col] = 0
            elif col in project_config.cat_features:
                sample_pandas_df[col] = "missing"
        
        # Basic cleaning for existing columns
        if col in project_config.num_features:
            sample_pandas_df[col] = pd.to_numeric(sample_pandas_df[col], errors='coerce').fillna(0)
        elif col in project_config.cat_features:
            sample_pandas_df[col] = sample_pandas_df[col].fillna("missing").astype(str)
            
    sample_input_payload_df = sample_pandas_df[all_expected_features]
    dataframe_records_payload = sample_input_payload_df.to_dict(orient="records")
    
    print("Sample records for prediction (first record if available):")
    if dataframe_records_payload:
        print(json.dumps(dataframe_records_payload[0], indent=2))
    else:
        print("No sample records generated.")

except Exception as e:
    print(f"Error preparing sample data: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Call the Endpoint

# COMMAND ----------
def call_endpoint(endpoint_url: str, databricks_token: str, data_payload: list):
    """
    Calls the model serving endpoint with a given input payload.
    Expects data_payload to be a list of records (dictionaries).
    """
    headers = {
        "Authorization": f"Bearer {databricks_token}",
        "Content-Type": "application/json",
    }
    # MLflow sklearn models served via Databricks expect this payload structure
    json_payload = {"dataframe_records": data_payload}

    response = requests.post(endpoint_url, headers=headers, json=json_payload)
    return response.status_code, response.text

# Construct the endpoint URL
serving_endpoint_url = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

print(f"Calling endpoint: {serving_endpoint_url}")

if dataframe_records_payload:
    # Call with the first sample record
    try:
        status_code, response_text = call_endpoint(
            endpoint_url=serving_endpoint_url,
            databricks_token=os.environ["DBR_TOKEN"],
            data_payload=[dataframe_records_payload[0]] # Send a list containing one record
        )
        print(f"\n--- Single Record Test ---")
        print(f"Response Status: {status_code}")
        try:
            print(f"Response JSON: {json.dumps(json.loads(response_text), indent=2)}")
        except json.JSONDecodeError:
            print(f"Response Text (not JSON): {response_text}")

        # Call with all sample records (if multiple)
        if len(dataframe_records_payload) > 1:
            status_code_batch, response_text_batch = call_endpoint(
                endpoint_url=serving_endpoint_url,
                databricks_token=os.environ["DBR_TOKEN"],
                data_payload=dataframe_records_payload 
            )
            print(f"\n--- Batch Records Test ({len(dataframe_records_payload)} records) ---")
            print(f"Response Status: {status_code_batch}")
            try:
                print(f"Response JSON: {json.dumps(json.loads(response_text_batch), indent=2)}")
            except json.JSONDecodeError:
                print(f"Response Text (not JSON): {response_text_batch}")

    except Exception as e:
        print(f"Error calling endpoint: {e}")
else:
    print("No sample data prepared, skipping endpoint call.")

# COMMAND ----------
print("Deployment notebook finished.")
# Optional Load Test section removed for brevity.