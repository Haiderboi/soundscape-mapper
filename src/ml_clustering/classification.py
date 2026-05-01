"""
ml_clustering/classification.py
----------------------------------
Soundscape-type classification, anomaly detection, and feature importance
analysis.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def train_soundscape_classifier(
    X: np.ndarray,
    y: np.ndarray | List,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    n_estimators: int = 100,
) -> Dict:
    """Train a Random Forest classifier for soundscape type prediction.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(n_samples, n_features)``.
    y:
        Target labels (strings or integers).
    test_size:
        Fraction of data reserved for testing.
    random_state:
        Reproducibility seed.
    cv_folds:
        Number of cross-validation folds.
    n_estimators:
        Number of trees in the random forest.

    Returns
    -------
    dict with keys:
        ``model``, ``scaler``, ``encoder``, ``accuracy``, ``cv_scores``,
        ``report``, ``confusion_matrix``, ``feature_importances``.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=test_size, random_state=random_state,
        stratify=y_enc,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(
        model, X_scaled, y_enc, cv=cv_folds, scoring="accuracy"
    )

    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)

    logger.info(
        "Classifier accuracy=%.4f  cv_mean=%.4f±%.4f",
        acc, cv_scores.mean(), cv_scores.std(),
    )
    return {
        "model": model,
        "scaler": scaler,
        "encoder": le,
        "accuracy": acc,
        "cv_scores": cv_scores,
        "report": report,
        "confusion_matrix": cm,
        "feature_importances": model.feature_importances_,
    }


def predict_soundscape_type(
    model,
    scaler,
    encoder,
    X: np.ndarray,
) -> np.ndarray:
    """Predict soundscape types for new observations.

    Returns
    -------
    np.ndarray of str
        Human-readable class labels.
    """
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return encoder.inverse_transform(y_pred)


def detect_anomalies(
    X: np.ndarray,
    contamination: float = 0.05,
    random_state: int = 42,
) -> np.ndarray:
    """Detect anomalous acoustic signatures using Isolation Forest.

    Parameters
    ----------
    X:
        Feature matrix.
    contamination:
        Expected proportion of anomalies in the dataset.
    random_state:
        Reproducibility seed.

    Returns
    -------
    np.ndarray of int
        Labels: ``1`` = normal, ``-1`` = anomaly.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        contamination=contamination, random_state=random_state
    )
    labels = iso.fit_predict(X_scaled)
    n_anomalies = int(np.sum(labels == -1))
    logger.info(
        "Anomaly detection: %d anomalies / %d total (contamination=%.2f)",
        n_anomalies, len(labels), contamination,
    )
    return labels


def feature_importance_analysis(
    model,
    feature_names: List[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Return a DataFrame of feature importances sorted by importance.

    Parameters
    ----------
    model:
        A fitted scikit-learn estimator with ``feature_importances_``.
    feature_names:
        Names corresponding to columns of the training feature matrix.
    top_n:
        Number of top features to return.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``importance``.
    """
    df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return df.head(top_n).reset_index(drop=True)
