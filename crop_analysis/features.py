"""Loading and feature engineering for the crop recommendation dataset."""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_TO_CLEAN = {
    "N": "Nitrogen",
    "P": "Phosphorous",
    "K": "Potassium",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "rainfall": "Rainfall",
    "ph": "pH",
    "label": "Crop_Type",
}

BASE_FEATURES = ["Temperature", "Humidity", "Rainfall", "pH", "Nitrogen", "Phosphorous", "Potassium"]
DERIVED_FEATURES = ["NPK_Avg", "Total_Nutrients", "Temp_Humidity_Index"]

IDEAL_TEMP_C = 27.0


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = set(RAW_TO_CLEAN) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = df.rename(columns=RAW_TO_CLEAN)
    df[BASE_FEATURES] = df[BASE_FEATURES].apply(pd.to_numeric, errors="raise")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise derived features. Nothing here looks at other rows, so it is
    safe to compute before the train/test split."""
    out = df.copy()
    out["Total_Nutrients"] = out["Nitrogen"] + out["Phosphorous"] + out["Potassium"]
    out["NPK_Avg"] = out["Total_Nutrients"] / 3
    out["Temp_Humidity_Index"] = out["Temperature"] * out["Humidity"] / 100
    return out


def temp_factor(temperature: pd.Series) -> pd.Series:
    """1.0 at the ideal temperature, falling linearly to 0 about 28 °C away."""
    return np.maximum(0, 1 - np.abs(temperature - IDEAL_TEMP_C) / (IDEAL_TEMP_C + 1))


# Coefficients of the synthetic yield target. Kept in one place so the SHAP
# analysis can compare what the model learned with the known ground truth.
YIELD_COEFS = {"NPK_Avg": 0.5, "Humidity": 0.2, "Rainfall": 0.1, "temp_factor": 5.0}


def synthetic_yield(df: pd.DataFrame) -> pd.Series:
    """The dataset has no yield column, so this project defines one.

    It is a deterministic function of the inputs. A model trained on it can
    only learn to approximate this formula; it does not predict real yield.
    """
    return (
        YIELD_COEFS["NPK_Avg"] * df["NPK_Avg"]
        + YIELD_COEFS["Humidity"] * df["Humidity"]
        + YIELD_COEFS["Rainfall"] * df["Rainfall"]
        + YIELD_COEFS["temp_factor"] * temp_factor(df["Temperature"])
    )


def fertility_level(total_nutrients: pd.Series, reference: pd.Series | None = None) -> pd.Series:
    """Low/Medium/High by tertiles of total N+P+K (of `reference` if given)."""
    ref = total_nutrients if reference is None else reference
    q1, q2 = ref.quantile([1 / 3, 2 / 3])
    return pd.cut(total_nutrients, [-np.inf, q1, q2, np.inf], labels=["Low", "Medium", "High"])
