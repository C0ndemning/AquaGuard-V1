from __future__ import annotations

import numpy as np
import pandas as pd

FLOW_COLUMNS = ["p227", "p235", "PUMP_1"]


def raw_sensor_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    pressure = sorted(
        [column for column in frame.columns if column.startswith("n")],
        key=lambda value: int(value[1:]) if value[1:].isdigit() else value,
    )
    flow = [column for column in FLOW_COLUMNS if column in frame.columns]
    return pressure, flow


def create_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the 205-feature hydraulic pipeline used by the deployed model.

    Important:
    - Input rows must be chronological.
    - Rolling/change features use only the current and earlier rows.
    - No calendar features are used.
    """
    working = frame.copy()

    if "Timestamp" in working.columns:
        working["Timestamp"] = pd.to_datetime(
            working["Timestamp"], errors="coerce"
        )
        working = working.sort_values("Timestamp").reset_index(drop=True)

    pressure_columns, flow_columns = raw_sensor_columns(working)
    raw_columns = pressure_columns + flow_columns

    if not pressure_columns:
        raise ValueError("No BattLeDIM pressure columns (n...) were found.")
    if len(flow_columns) != 3:
        missing = sorted(set(FLOW_COLUMNS) - set(flow_columns))
        raise ValueError(f"Missing required flow columns: {missing}")

    numeric = working[raw_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.ffill().bfill().fillna(0.0)

    feature_data: dict[str, pd.Series] = {
        column: numeric[column].astype(float) for column in raw_columns
    }

    pressure_data = numeric[pressure_columns].astype(float)
    pressure_mean = pressure_data.mean(axis=1)
    pressure_std = pressure_data.std(axis=1)
    pressure_min = pressure_data.min(axis=1)
    pressure_max = pressure_data.max(axis=1)
    pressure_median = pressure_data.median(axis=1)
    pressure_q25 = pressure_data.quantile(0.25, axis=1)
    pressure_q75 = pressure_data.quantile(0.75, axis=1)

    feature_data["pressure_mean"] = pressure_mean
    feature_data["pressure_std"] = pressure_std
    feature_data["pressure_min"] = pressure_min
    feature_data["pressure_max"] = pressure_max
    feature_data["pressure_range"] = pressure_max - pressure_min
    feature_data["pressure_median"] = pressure_median
    feature_data["pressure_q25"] = pressure_q25
    feature_data["pressure_q75"] = pressure_q75
    feature_data["pressure_iqr"] = pressure_q75 - pressure_q25

    flow_data = numeric[flow_columns].astype(float)
    flow_total = flow_data.sum(axis=1)
    flow_mean = flow_data.mean(axis=1)
    flow_std = flow_data.std(axis=1)
    flow_min = flow_data.min(axis=1)
    flow_max = flow_data.max(axis=1)

    feature_data["flow_total"] = flow_total
    feature_data["flow_mean"] = flow_mean
    feature_data["flow_std"] = flow_std
    feature_data["flow_min"] = flow_min
    feature_data["flow_max"] = flow_max
    feature_data["flow_range"] = flow_max - flow_min

    for first_index in range(len(flow_columns)):
        for second_index in range(first_index + 1, len(flow_columns)):
            first_sensor = flow_columns[first_index]
            second_sensor = flow_columns[second_index]
            feature_data[
                f"flow_difference_{first_sensor}_minus_{second_sensor}"
            ] = numeric[first_sensor] - numeric[second_sensor]

    for column in pressure_columns:
        feature_data[f"{column}_deviation_from_mean"] = (
            numeric[column] - pressure_mean
        )

    for column in raw_columns:
        values = numeric[column].astype(float)
        feature_data[f"{column}_change"] = values.diff().fillna(0.0)
        feature_data[f"{column}_rolling_mean_3"] = (
            values.rolling(3, min_periods=1).mean()
        )
        feature_data[f"{column}_rolling_std_3"] = (
            values.rolling(3, min_periods=1).std().fillna(0.0)
        )

    feature_data["pressure_mean_change"] = pressure_mean.diff().fillna(0.0)
    pressure_range = feature_data["pressure_range"]
    feature_data["pressure_range_change"] = pressure_range.diff().fillna(0.0)
    feature_data["pressure_mean_rolling_3"] = (
        pressure_mean.rolling(3, min_periods=1).mean()
    )
    feature_data["pressure_mean_rolling_6"] = (
        pressure_mean.rolling(6, min_periods=1).mean()
    )
    feature_data["pressure_std_rolling_3"] = (
        pressure_std.rolling(3, min_periods=1).mean()
    )

    feature_data["flow_total_change"] = flow_total.diff().fillna(0.0)
    feature_data["flow_total_rolling_3"] = (
        flow_total.rolling(3, min_periods=1).mean()
    )
    feature_data["flow_total_rolling_6"] = (
        flow_total.rolling(6, min_periods=1).mean()
    )

    epsilon = 1e-6
    feature_data["flow_to_pressure_ratio"] = (
        flow_total / (pressure_mean.abs() + epsilon)
    )
    feature_data["pressure_drop_per_flow"] = (
        pressure_range / (flow_total.abs() + epsilon)
    )

    features = pd.DataFrame(feature_data, index=working.index)
    return (
        features.replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
        .astype(np.float32)
    )


def prepare_model_matrix(
    raw_frame: pd.DataFrame,
    saved_features: list[str],
) -> pd.DataFrame:
    engineered = create_engineered_features(raw_frame)
    missing = [name for name in saved_features if name not in engineered.columns]
    if missing:
        raise ValueError(
            "The feature pipeline could not recreate model inputs: "
            + ", ".join(missing[:20])
        )
    matrix = engineered.loc[:, saved_features].copy()
    if list(matrix.columns) != list(saved_features):
        raise RuntimeError("Feature ordering failed.")
    return matrix
