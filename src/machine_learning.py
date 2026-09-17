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
from sklearn.model_selection import TimeSeriesSplit
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


# ============================================================================
# Walk-Forward / Expanding-Window Cross-Validation (Issue #45 item 7)
# ============================================================================
# The final chronological train/test split (time_series_split() above) stays
# the UNTOUCHED headline out-of-sample holdout -- it is NEVER seen by any
# fold below. This section only adds expanding-window TimeSeriesSplit
# validation INSIDE the pre-holdout training region, to measure how stable
# a model's accuracy/ROC AUC are across several chronological, expanding
# train/validation windows -- fold dispersion measures temporal instability,
# it does NOT prove future predictability.

DEFAULT_CV_FOLDS = 5
MIN_CV_FOLDS = 2
MIN_FOLD_TRAIN_SIZE = 15
MIN_FOLD_VAL_SIZE = 5


def _fit_and_score_fold(model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Fit ONE model on a single fold's training rows only (the
    LogisticRegression scaler is fit fresh here, on this fold's training
    data only -- never on the validation rows or on any other fold's data)
    and score it on that fold's validation rows. Returns
    {"accuracy": float, "baseline_accuracy": float, "roc_auc": float or None}.
    Raises if the model genuinely cannot be fit (e.g. a single class present
    in y_train) -- callers must catch this and skip the fold rather than
    fabricate a score.
    """
    if model_type == "Logistic Regression":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.fillna(0))
        X_val_scaled = scaler.transform(X_val.fillna(0))
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]
    else:
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=10,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train.fillna(0), y_train)
        y_pred = model.predict(X_val.fillna(0))
        y_prob = model.predict_proba(X_val.fillna(0))[:, 1]

    accuracy = float(accuracy_score(y_val, y_pred))
    baseline_accuracy = _baseline_majority_class_accuracy(y_train, y_val)
    roc_auc = None
    if len(set(y_val)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_val, y_prob))
        except Exception:
            roc_auc = None
    return {"accuracy": accuracy, "baseline_accuracy": baseline_accuracy, "roc_auc": roc_auc}


def _feasible_fold_count(n: int, n_splits: int) -> bool:
    """Whether TimeSeriesSplit(n_splits) on `n` pre-holdout rows produces
    folds that ALL meet the minimum train/validation size thresholds
    above -- checked directly against the actual split, not estimated, so
    this can never disagree with what walk_forward_validation() actually
    does below."""
    if n < n_splits + 1:
        return False
    tss = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tss.split(np.zeros(n)):
        if len(train_idx) < MIN_FOLD_TRAIN_SIZE or len(val_idx) < MIN_FOLD_VAL_SIZE:
            return False
    return True


def _choose_n_splits(n: int, target_splits: int = DEFAULT_CV_FOLDS,
                      min_splits: int = MIN_CV_FOLDS):
    """The largest feasible fold count from `min_splits` up to
    `target_splits` (target 5, reduced safely when the pre-holdout region
    is too small) -- or None if even `min_splits` folds aren't feasible."""
    for k in range(min(target_splits, n - 1), min_splits - 1, -1):
        if k >= min_splits and _feasible_fold_count(n, k):
            return k
    return None


def walk_forward_validation(X_train: pd.DataFrame, y_train: pd.Series, model_type: str,
                             target_splits: int = DEFAULT_CV_FOLDS) -> dict:
    """Expanding-window TimeSeriesSplit validation over the PRE-HOLDOUT
    training region only (`X_train`/`y_train` -- the final chronological
    test split is never passed to this function, so it can never leak into
    a CV fold). Each fold is a chronological, expanding, non-overlapping
    train -> validation split (TimeSeriesSplit's own guarantee); no
    shuffling is used anywhere.

    Returns, when at least MIN_CV_FOLDS folds could be fit:
        {"available": True, "n_folds": int, "folds": [ {fold, train_start,
         train_end, val_start, val_end, n_train, n_val, accuracy,
         baseline_accuracy, roc_auc}, ... ],
         "accuracy_mean": float, "accuracy_std": float,
         "roc_auc_mean": float or None, "roc_auc_std": float or None,
         "n_valid_auc_folds": int}
    or, when there isn't enough pre-holdout data for at least MIN_CV_FOLDS
    valid chronological folds:
        {"available": False, "reason": str}
    Never fabricates statistics for a fold that couldn't actually be fit
    (e.g. a single class in that fold's training rows) -- such folds are
    skipped, not scored as 0/NaN.
    """
    n = len(X_train)
    n_splits = _choose_n_splits(n, target_splits, MIN_CV_FOLDS)
    if n_splits is None:
        return {
            "available": False,
            "reason": (
                f"insufficient pre-holdout data ({n} row(s)) for at least "
                f"{MIN_CV_FOLDS} chronological expanding-window folds "
                f"(each fold needs >= {MIN_FOLD_TRAIN_SIZE} training and "
                f">= {MIN_FOLD_VAL_SIZE} validation observations)"
            ),
        }

    tss = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for fold_num, (train_idx, val_idx) in enumerate(tss.split(X_train), start=1):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        try:
            score = _fit_and_score_fold(model_type, X_tr, y_tr, X_val, y_val)
        except Exception:
            # A fold that genuinely cannot be fit (e.g. a single class in
            # this fold's training rows) is skipped entirely -- never
            # scored as a fabricated 0/NaN result.
            continue
        folds.append({
            "fold": fold_num,
            "train_start": str(X_tr.index.min().date()), "train_end": str(X_tr.index.max().date()),
            "val_start": str(X_val.index.min().date()), "val_end": str(X_val.index.max().date()),
            "n_train": len(X_tr), "n_val": len(X_val),
            "accuracy": score["accuracy"], "baseline_accuracy": score["baseline_accuracy"],
            "roc_auc": score["roc_auc"],
        })

    if len(folds) < MIN_CV_FOLDS:
        return {
            "available": False,
            "reason": (
                f"fewer than {MIN_CV_FOLDS} folds could actually be fit "
                "(e.g. a fold's training rows contained only one class)"
            ),
        }

    accuracies = [f["accuracy"] for f in folds]
    aucs = [f["roc_auc"] for f in folds if f["roc_auc"] is not None]
    return {
        "available": True,
        "n_folds": len(folds),
        "folds": folds,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies, ddof=0)),
        "roc_auc_mean": float(np.mean(aucs)) if aucs else None,
        "roc_auc_std": float(np.std(aucs, ddof=0)) if aucs else None,
        "n_valid_auc_folds": len(aucs),
    }


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
        # Walk-forward validation (Issue #45 item 7): expanding-window
        # TimeSeriesSplit folds over X_train/y_train ONLY -- the final
        # holdout (X_test/y_test) above is never passed in, so it can never
        # leak into a CV fold or be used for tuning/CV summaries.
        result["walk_forward_cv"] = walk_forward_validation(X_train, y_train, model_type)
        result["disclaimer"] = DISCLAIMER
        result["error"] = None
        return result

    except Exception as e:
        return {"error": f"Model training failed: {str(e)}"}
