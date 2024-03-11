"""Unsupervised anomaly detection."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from typing import Tuple


def detect_anomalies_iforest(
    X: np.ndarray,
    contamination: float = 0.02,
    n_estimators: int = 200,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect anomalies using Isolation Forest.
    
    Args:
        X: Feature matrix
        contamination: Expected proportion of anomalies
        n_estimators: Number of trees
        random_state: Random seed
    
    Returns:
        Tuple of (anomaly_flags, anomaly_scores)
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    
    predictions = model.fit_predict(X)
    scores = -model.score_samples(X)  # Negative for higher = more anomalous
    
    anomalies = predictions == -1
    
    return anomalies, scores


def detect_anomalies_ocsvm(
    X: np.ndarray,
    nu: float = 0.02,
    kernel: str = "rbf",
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect anomalies using One-Class SVM.
    
    Args:
        X: Feature matrix
        nu: Upper bound on fraction of outliers
        kernel: Kernel type
    
    Returns:
        Tuple of (anomaly_flags, anomaly_scores)
    """
    model = OneClassSVM(nu=nu, kernel=kernel)
    
    predictions = model.fit_predict(X)
    scores = -model.decision_function(X)  # Negative for higher = more anomalous
    
    anomalies = predictions == -1
    
    return anomalies, scores
