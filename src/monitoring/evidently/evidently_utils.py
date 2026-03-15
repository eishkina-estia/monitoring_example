import os
import pickle

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from src import common as common
DATA_MONITORING_REF_PATH = common.CONFIG['paths']['monitoring']['data_reference']

COL_NAME_TARGET_TRUE = "target"
COL_NAME_TARGET_PRED = "prediction"

def build_reference_dataframe():
    """
    Load reference data and build the reference dataframe for Evidently.

    Expected reference data format (pickled dictionary):
    - X_ref: feature values (pd.DataFrame)
    - y_ref_true: true target values (pd.Series)
    - y_ref_pred: values predicted by the model being monitored (pd.Series)

    Returns
    -------
    ref_df : pd.DataFrame
        Reference dataset for Evidently.
    feature_columns : list[str]
        Ordered list of feature columns.
    """

    with open(DATA_MONITORING_REF_PATH, "rb") as file:
        ref_data_dict = pickle.load(file)

    X_ref = ref_data_dict["X_ref"]
    y_ref_true = ref_data_dict["y_ref_true"].rename(COL_NAME_TARGET_TRUE)
    y_ref_pred = ref_data_dict["y_ref_pred"].rename(COL_NAME_TARGET_PRED)

    ref_df = pd.concat([X_ref, y_ref_true, y_ref_pred], axis=1)
    feature_columns = list(X_ref.columns)

    return ref_df, feature_columns


def load_current_dataframe(csv_path: str, feature_columns: list[str]) -> pd.DataFrame:
    """
    Load current data from CSV.

    Convention:
    - all columns except the last one are features
    - the last column is the true target
    - feature order must match the reference exactly
    """
    current_df = pd.read_csv(csv_path)

    if current_df.shape[1] < 2:
        raise ValueError(
            "Current dataset must contain at least one feature column and one target column."
        )

    current_feature_cols = list(current_df.columns[:-1])

    if current_feature_cols != feature_columns:
        raise ValueError(
            "Feature columns in the current dataset do not match the reference order.\n"
            f"Expected: {feature_columns}\n"
            f"Got: {current_feature_cols}"
        )

    current_df.columns = feature_columns + [COL_NAME_TARGET_TRUE]

    return current_df



def generate_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, output_path: str):
    """
    Build and save an Evidently drift report.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Reference dataset used for drift comparison.
    current_df : pd.DataFrame
        Current dataset to evaluate.
    output_path : str
        Path where the HTML report will be saved.
    """

    # If the current dataframe does not contain prediction values, remove the
    # prediction column from the reference dataframe as well.
    if COL_NAME_TARGET_PRED not in current_df.columns and COL_NAME_TARGET_PRED in reference_df.columns:
        reference_df = reference_df.drop(columns=[COL_NAME_TARGET_PRED])

    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    eval_result = report.run(reference_data=reference_df, current_data=current_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    eval_result.save_html(output_path)

    print(f"Report generated: {output_path}")

