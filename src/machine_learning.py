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


def _baseline_majority_class_accuracy(y_train: pd.Series, y_test: pd.Series) -> float:
    """Simple baseline: always predict whichever class (up/down) was more
    frequent in the TRAINING set (never peeking at y_test's own balance --
    that would leak test information into the baseline itself). A trained
    model that cannot beat this on the same held-out test set is providing
    no directional information beyond that period's own class balance, so
    this is computed and surfaced alongside every real model's accuracy
    rather than left as an abstract "~50% for a coin flip" claim in prose.
    """
    majority_class = 1 if len(y_train) and y_train.mean() >= 0.5 else 0
    baseline_pred = np.full(len(y_test), majority_class)
    return float(accuracy_score(y_test, baseline_pred))


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


def roc_auc_interpretation_level(roc_auc: float) -> str:
    """Bucket a ROC AUC value into "below_chance"/"weak"/"moderate" for the
    Results page's dynamic interpretation (Issue #41 item H). Before this,
    only a narrow +-0.05-around-0.5 band ("weak") had any interpretation at
    all -- a result like 0.4356 fell outside it and received none. Every
    value now gets one:

    - "below_chance": AUC < 0.50 -- the model ranked positive vs negative
      observations worse than chance on THIS held-out window.
    - "weak": 0.50 <= AUC <= 0.55 -- little to no discriminatory power,
      close to random guessing.
    - "moderate": AUC > 0.55 -- some ranking ability, but still just one
      test window, not a stable predictive edge.

    Simple PRODUCT UI buckets, not an academic or regulatory AUC standard --
    same pattern as src.financial_metrics.concentration_level().
    """
    if roc_auc < 0.50:
        return "below_chance"
    if roc_auc <= 0.55:
        return "weak"
    return "moderate"


CONFUSION_MATRIX_SKEW_THRESHOLD = 0.65


def confusion_matrix_skew_direction(predicted_up: int, total: int,
                                     threshold: float = CONFUSION_MATRIX_SKEW_THRESHOLD):
    """Whether the model's predictions on the held-out test set are
    materially skewed toward one direction (Issue #41 item I). Returns
    "up"/"down" when `predicted_up / total` (or its complement) is at or
    above `threshold`, else None (no material skew). `total` <= 0 always
    returns None rather than dividing by zero."""
    if total <= 0:
        return None
    frac_up = predicted_up / total
    if frac_up >= threshold:
        return "up"
    if (1 - frac_up) >= threshold:
        return "down"
    return None


def run_ml_pipeline(prices: pd.Series, volume: pd.Series = None,
                     model_type: str = "Random Forest",
                     test_size: float = 0.2, lookahead: int = 1) -> dict:
    """
    Full ML pipeline: prepare data, split, train, evaluate.
    Returns results dict with metrics, feature importance, and predictions.
    """
    X, y, index = prepare_ml_dataset(prices, volume, lookahead=lookahead)

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
        # Explicit train/test window disclosure (M5 -- issue #18 Stage 5):
        # the page previously only showed observation COUNTS, never the
        # actual as-of date range the "out-of-sample" test metrics apply
        # to, which is required to judge whether a result is even current.
        result["train_start"] = str(X_train.index.min().date())
        result["train_end"] = str(X_train.index.max().date())
        result["test_start"] = str(X_test.index.min().date())
        result["test_end"] = str(X_test.index.max().date())
        # Target definition, made explicit and derived from the actual
        # `lookahead` argument rather than a hardcoded "next-day" string, so
        # this can never silently drift from what prepare_ml_dataset() (and
        # the label it built) actually computed.
        result["lookahead_periods"] = lookahead
        # Simple baseline (M5): a model that cannot beat "always predict the
        # training set's majority class" on the SAME held-out test set is
        # not demonstrating real directional skill for this ETF/period.
        result["baseline_accuracy"] = round(_baseline_majority_class_accuracy(y_train, y_test), 4)
        result["disclaimer"] = DISCLAIMER
        result["error"] = None
        return result

    except Exception as e:
        return {"error": f"Model training failed: {str(e)}"}
