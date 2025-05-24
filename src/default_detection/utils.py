"""Utility class."""

import numpy as np
import pandas as pd


def serving_pred_function(client_ids: list, banned_client_list: pd.DataFrame, predictions: list[float]) -> list[float]:
    """Adjust predictions: if a client is on the banned list, set their prediction to 1.

    :param client_ids: Array of client IDs corresponding to the predictions.
    :param banned_client_list: DataFrame containing a column 'banned_clients_ids' with banned client IDs.
    :param predictions: Array of model predictions.
    :return: Adjusted predictions array, where banned clients have prediction set to 1.
    """
    banned_ids = set(banned_client_list["banned_clients_ids"].values)
    adjusted = [
        1.0 if client_id in banned_ids else pred for client_id, pred in zip(client_ids, predictions, strict=True)
    ]  # Ensure float for consistency
    return adjusted


def adjust_probabilities_for_high_risk(
    model_input_df: pd.DataFrame,
    probabilities: np.ndarray,  # 1D array of probabilities for the positive class
    high_risk_artifact_df: pd.DataFrame,
    client_id_col_in_input: str,
    client_id_col_in_artifact: str,
    adjustment_factor: float = 0.2,
    cap_value: float = 1.0,
) -> np.ndarray:
    """Adjust probabilities for clients identified as high risk.

    If a client_id from model_input_df is in high_risk_artifact_df,
    their probability is increased by adjustment_factor, capped at cap_value.

    :param model_input_df: DataFrame containing client identifiers for the current batch.
    :param probabilities: NumPy array of initial predicted probabilities for the positive class.
    :param high_risk_artifact_df: DataFrame loaded from the artifact defining high-risk clients.
    :param client_id_col_in_input: Name of the client ID column in model_input_df.
    :param client_id_col_in_artifact: Name of the client ID column in high_risk_artifact_df.
    :param adjustment_factor: The value to add to the probability for high-risk clients.
    :param cap_value: The maximum value a probability can reach after adjustment.
    :return: NumPy array of adjusted probabilities.
    """
    if client_id_col_in_input not in model_input_df.columns:
        raise ValueError(
            f"Client ID column '{client_id_col_in_input}' not found in model input. "
            f"Available columns: {model_input_df.columns.tolist()}"
        )
    if client_id_col_in_artifact not in high_risk_artifact_df.columns:
        raise ValueError(
            f"Client ID column '{client_id_col_in_artifact}' not found in high-risk artifact. "
            f"Available columns: {high_risk_artifact_df.columns.tolist()}"
        )

    high_risk_ids = set(high_risk_artifact_df[client_id_col_in_artifact].unique())
    adjusted_probabilities = probabilities.copy()

    # Ensure client_ids from input are of a comparable type to those in high_risk_ids
    # For example, if one is int and other is str, comparison will fail.
    # This is a common source of issues, so defensive type casting or checking might be needed
    # depending on the actual data. For now, assuming types are compatible.

    for i, client_id in enumerate(model_input_df[client_id_col_in_input]):
        if client_id in high_risk_ids:
            adjusted_probabilities[i] = min(adjusted_probabilities[i] + adjustment_factor, cap_value)
    return adjusted_probabilities
