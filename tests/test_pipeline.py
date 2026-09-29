import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crop_analysis import features, recommend, yield_model  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return features.add_features(features.load(ROOT / "Crop_recommendation.csv"))


def test_load_renames_and_validates(df):
    assert set(features.BASE_FEATURES) <= set(df.columns)
    assert df["Crop_Type"].nunique() == 22
    assert len(df) == 2200


def test_load_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"N": [1], "P": [2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        features.load(p)


def test_derived_features_are_row_wise(df):
    row = df.iloc[0]
    assert row["Total_Nutrients"] == row["Nitrogen"] + row["Phosphorous"] + row["Potassium"]
    assert row["NPK_Avg"] == pytest.approx(row["Total_Nutrients"] / 3)
    # computing on a subset gives identical values -> no cross-row leakage
    sub = features.add_features(df.drop(columns=features.DERIVED_FEATURES).iloc[:5])
    pd.testing.assert_frame_equal(sub[features.DERIVED_FEATURES], df[features.DERIVED_FEATURES].iloc[:5])


def test_temp_factor_peaks_at_ideal():
    t = pd.Series([27.0, 13.0, 55.0, -1.0])
    np.testing.assert_allclose(features.temp_factor(t), [1.0, 0.5, 0.0, 0.0])


def test_synthetic_yield_matches_formula(df):
    r = df.iloc[10]
    expected = (0.5 * r["NPK_Avg"] + 0.2 * r["Humidity"] + 0.1 * r["Rainfall"]
                + 5 * max(0, 1 - abs(r["Temperature"] - 27) / 28))
    assert features.synthetic_yield(df).iloc[10] == pytest.approx(expected)


def test_fertility_levels_are_tertiles(df):
    counts = features.fertility_level(df["Total_Nutrients"]).value_counts(normalize=True)
    assert set(counts.index) == {"Low", "Medium", "High"}
    assert counts.min() > 0.28


def test_ground_truth_ranking(df):
    spread = yield_model.ground_truth_spread(df)
    assert spread.index[0].startswith("NPK")
    assert spread.index[-1].startswith("Temperature")


def test_recommend_returns_sorted_probabilities(df):
    model = recommend.candidates()["random_forest"].fit(df[features.BASE_FEATURES], df["Crop_Type"].astype("category").cat.codes)
    enc = recommend.LabelEncoder().fit(df["Crop_Type"])
    rice = df[df["Crop_Type"] == "rice"].iloc[0][features.BASE_FEATURES].to_dict()
    top = recommend.recommend(model, enc, rice, top=3)
    assert top[0][0] == "rice"
    assert [p for _, p in top] == sorted((p for _, p in top), reverse=True)
