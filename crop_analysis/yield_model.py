"""XGBoost regression on the synthetic yield target, with SHAP explanations."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from .features import BASE_FEATURES, DERIVED_FEATURES, YIELD_COEFS, synthetic_yield, temp_factor

NUMERIC = BASE_FEATURES + DERIVED_FEATURES
CATEGORICAL = ["Crop_Type"]

PARAM_GRID = {
    "model__n_estimators": [100, 200, 400],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5],
}


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


def train(df: pd.DataFrame, seed: int = 42) -> dict:
    """Tune XGBoost with CV on the training split; report held-out metrics
    for XGBoost and a linear baseline."""
    X = df[NUMERIC + CATEGORICAL]
    y = synthetic_yield(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=df["Crop_Type"]
    )

    xgb = Pipeline([("prep", _preprocessor()), ("model", XGBRegressor(random_state=seed, n_jobs=-1))])
    search = GridSearchCV(xgb, PARAM_GRID, cv=KFold(5, shuffle=True, random_state=seed),
                          scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train)

    # The target is linear except for the temperature term, so a linear model
    # is the honest baseline: it shows how much of XGBoost's score is simply
    # recovering a known formula.
    linear = Pipeline([("prep", _preprocessor()), ("model", LinearRegression())]).fit(X_train, y_train)

    return {
        "model": search.best_estimator_,
        "best_params": {k.removeprefix("model__"): v for k, v in search.best_params_.items()},
        "cv_r2": round(float(search.best_score_), 4),
        "test": {
            "xgboost": _metrics(y_test, search.predict(X_test)),
            "linear_baseline": _metrics(y_test, linear.predict(X_test)),
        },
        "X_train": X_train,
        "X_test": X_test,
    }


def shap_importance(model: Pipeline, X: pd.DataFrame, max_rows: int = 1000, seed: int = 0):
    """Mean |SHAP| per feature, using the *fitted* preprocessor from the pipeline."""
    import shap

    prep, reg = model.named_steps["prep"], model.named_steps["model"]
    sample = X.sample(min(max_rows, len(X)), random_state=seed)
    Xt = pd.DataFrame(prep.transform(sample), columns=prep.get_feature_names_out(), index=sample.index)
    explanation = shap.TreeExplainer(reg)(Xt)
    importance = (
        pd.Series(np.abs(explanation.values).mean(axis=0), index=Xt.columns)
        .sort_values(ascending=False)
    )
    return importance, explanation


def ground_truth_spread(df: pd.DataFrame) -> pd.Series:
    """Mean absolute deviation of each term of the known target formula.

    For an additive model this is exactly what mean |SHAP| of that term would
    be, so it is the reference to check the SHAP ranking against.
    """
    terms = {
        "NPK (0.5 × NPK_Avg)": YIELD_COEFS["NPK_Avg"] * df["NPK_Avg"],
        "Humidity (0.2 ×)": YIELD_COEFS["Humidity"] * df["Humidity"],
        "Rainfall (0.1 ×)": YIELD_COEFS["Rainfall"] * df["Rainfall"],
        "Temperature (5 × temp_factor)": YIELD_COEFS["temp_factor"] * temp_factor(df["Temperature"]),
    }
    return pd.Series({k: float((v - v.mean()).abs().mean()) for k, v in terms.items()}).sort_values(
        ascending=False
    )


# Which model features belong to which term of the known formula. Nutrient
# features are collinear (Total = N + P + K, Avg = Total / 3), so SHAP
# spreads the NPK credit across all five; summing them makes it comparable.
SHAP_GROUPS = {
    "NPK (0.5 × NPK_Avg)": ["Nitrogen", "Phosphorous", "Potassium", "NPK_Avg", "Total_Nutrients"],
    "Humidity (0.2 ×)": ["Humidity"],
    "Rainfall (0.1 ×)": ["Rainfall"],
    "Temperature (5 × temp_factor)": ["Temperature"],
}


def grouped_shap(importance: pd.Series) -> pd.Series:
    grouped = {g: float(importance.reindex(cols).fillna(0).sum()) for g, cols in SHAP_GROUPS.items()}
    used = {c for cols in SHAP_GROUPS.values() for c in cols}
    grouped["Temp × Humidity index"] = float(importance.get("Temp_Humidity_Index", 0.0))
    grouped["pH + crop type (not in formula)"] = float(
        importance.drop(labels=[*used, "Temp_Humidity_Index"], errors="ignore").sum())
    return pd.Series(grouped)
