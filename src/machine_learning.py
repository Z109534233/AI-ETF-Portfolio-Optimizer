"""
Machine Learning Module
Educational demonstration of ML models for ETF direction prediction.
Uses time-series-aware train/test splitting to prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from src.technical_indicators import create_ml_features


DISCLAIMER = (
    "Market predictions are uncertain and are provided only as an educational demonstration. "
    "Past performance does not guarantee future results. "
    "This tool is not financial advice."
)


def prepare_ml_dataset(prices: pd.Series, volume: pd.Series = None,
                        lookahead: int = 1) -> tuple:
    """
    Prepare features and labels for ML classification.
    Label: 1 if next-period return > 0, else 0.
    Uses time-series split (no shuffling).
    """
    features_df = create_ml_features(prices, volume)

    # Target: next-day direction
    future_return = prices.pct_change(lookahead).shift(-lookahead)
    labels = (future_return > 0).astype(int)

    # Align
    combined = features_df.join(labels.rename("Target")).dropna()
    if len(combined) < 50:
        return None, None, None

    X = combined.drop(columns=["Target"])
    y = combined["Target"]

    return X, y, combined.index


def time_series_split(X: pd.DataFrame, y: pd.Series,
                       test_size: float = 0.2) -> tuple:
    """
    Time-series aware train/test split.
    No shuffling to prevent data leakage.
    """
    n = len(X)
    split_idx = int(n * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Train and evaluate Logistic Regression model."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(0))
    X_test_scaled = scaler.transform(X_test.fillna(0))

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = _compute_metrics(y_test, y_pred, y_prob)
    feature_importance = pd.Series(
        np.abs(model.coef_[0]),
        index=X_train.columns
    ).sort_values(ascending=False)

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_test": y_test,
        "model_name": "Logistic Regression",
    }


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series,
                         n_estimators: int = 100) -> dict:
    """Train and evaluate Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train.fillna(0), y_train)

    y_pred = model.predict(X_test.fillna(0))
    y_prob = model.predict_proba(X_test.fillna(0))[:, 1]

    metrics = _compute_metrics(y_test, y_pred, y_prob)
    feature_importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    return {
        "model": model,
        "scaler": None,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_test": y_test,
        "model_name": "Random Forest",
    }


def _compute_metrics(y_test, y_pred, y_prob) -> dict:
    """Compute classification metrics."""
    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    try:
        metrics["ROC AUC"] = round(roc_auc_score(y_test, y_prob), 4)
    except Exception:
        metrics["ROC AUC"] = "N/A"

    cm = confusion_matrix(y_test, y_pred)
    metrics["Confusion Matrix"] = cm
    return metrics


def run_ml_pipeline(prices: pd.Series, volume: pd.Series = None,
                     model_type: str = "Random Forest",
                     test_size: float = 0.2) -> dict:
    """
    Full ML pipeline: prepare data, split, train, evaluate.
    Returns results dict with metrics, feature importance, and predictions.
    """
    X, y, index = prepare_ml_dataset(prices, volume)

    if X is None:
        return {"error": "Insufficient data for ML analysis. Need at least 50 observations."}

    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size)

    if len(X_train) < 20 or len(X_test) < 10:
        return {"error": "Not enough data for train/test split. Try a longer date range."}

    try:
        if model_type == "Logistic Regression":
            result = train_logistic_regression(X_train, y_train, X_test, y_test)
        else:
            result = train_random_forest(X_train, y_train, X_test, y_test)

        result["train_size"] = len(X_train)
        result["test_size"] = len(X_test)
        result["test_index"] = X_test.index
        result["disclaimer"] = DISCLAIMER
        result["error"] = None
        return result

    except Exception as e:
        return {"error": f"Model training failed: {str(e)}"}
