"""Crop recommendation: predict the best-suited crop from soil and weather."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from .features import BASE_FEATURES


def candidates(seed: int = 42) -> dict:
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                 random_state=seed, n_jobs=-1),
    }


def compare(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """5-fold stratified CV on the training split, then one held-out test.

    Only the seven raw measurements are used as inputs: the engineered
    features are linear combinations of them and add nothing for trees.
    """
    X = df[BASE_FEATURES]
    enc = LabelEncoder().fit(df["Crop_Type"])
    y = enc.transform(df["Crop_Type"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows, fitted = [], {}
    for name, model in candidates(seed).items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "cv_accuracy": np.mean(scores["test_accuracy"]),
            "cv_accuracy_std": np.std(scores["test_accuracy"]),
            "cv_macro_f1": np.mean(scores["test_f1_macro"]),
            "test_accuracy": accuracy_score(y_test, pred),
            "test_macro_f1": f1_score(y_test, pred, average="macro"),
        })
        fitted[name] = model

    table = pd.DataFrame(rows).set_index("model").round(4).sort_values("cv_macro_f1", ascending=False)
    return table, {"models": fitted, "encoder": enc, "X_test": X_test, "y_test": y_test}


def recommend(model, encoder: LabelEncoder, sample: dict, top: int = 3) -> list[tuple[str, float]]:
    """Top-k crops with probabilities for one field measurement."""
    X = pd.DataFrame([sample])[BASE_FEATURES]
    probs = model.predict_proba(X)[0]
    order = np.argsort(probs)[::-1][:top]
    return [(str(encoder.classes_[i]), float(probs[i])) for i in order]
