"""
Machine learning model module for Retain.
Builds scikit-learn pipelines, trains models, evaluates metrics,
computes permutation importance, and generates out-of-fold probabilities.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def build_pipeline(classifier, num_cols, cat_cols) -> Pipeline:
    """
    Creates an end-to-end scikit-learn Pipeline with ColumnTransformer:
    - Scales numeric columns using StandardScaler.
    - One-hot encodes categorical columns using OneHotEncoder.
    - Preserves raw input column names for seamless inference.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    return pipe


def get_models():
    """
    Returns exactly 3 models: LogisticRegression, RandomForestClassifier, and XGBClassifier.
    Unweighted classes preserve calibrated posterior probabilities.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_split=10,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }


def evaluate_pipeline(pipeline, X_test, y_test):
    """
    Calculates key performance metrics on the test set:
    - ROC-AUC, Accuracy, Precision, Recall, F1 (at 0.5 threshold)
    - Confusion matrix
    - Calibration curve with 10 uniform bins
    - ROC curve coordinates (fpr, tpr)
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_prob))
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    # Downsample ROC points for lightweight JSON serialization
    step = max(1, len(fpr) // 50)
    roc_points = {
        "fpr": [round(float(v), 4) for v in fpr[::step]],
        "tpr": [round(float(v), 4) for v in tpr[::step]]
    }
    if roc_points["fpr"][-1] != 1.0:
        roc_points["fpr"].append(1.0)
        roc_points["tpr"].append(1.0)

    return {
        "roc_auc": round(roc_auc, 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm,
        "calibration": {
            "prob_true": [round(float(v), 4) for v in prob_true],
            "prob_pred": [round(float(v), 4) for v in prob_pred]
        },
        "roc_curve": roc_points
    }


def calibrate_model(pipeline, X_train, y_train, cv: int = 5):
    """
    Calibrates model predicted probabilities using CalibratedClassifierCV with sigmoid (Platt scaling).
    """
    calibrated = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=cv)
    calibrated.fit(X_train, y_train)
    return calibrated


def compute_permutation_importance(pipeline, X_test, y_test, n_repeats: int = 10, random_state: int = 42):
    """
    Measures the top drivers of churn using permutation feature importance.
    """
    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state
    )

    feature_names = list(X_test.columns)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    return importance_df


def generate_out_of_fold_probabilities(pipeline, X, y, cv: int = 5, random_state: int = 42) -> np.ndarray:
    """
    Computes out-of-fold predicted probabilities for every customer using 5-fold cross-validation.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    oof_probs = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")[:, 1]
    return oof_probs


def save_artifacts(model, metrics_dict, model_path: str, metrics_path: str):
    """
    Saves the trained pipeline with joblib and metrics as JSON.
    """
    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)


def load_artifacts(model_path: str, metrics_path: str):
    """
    Loads saved pipeline and metrics from disk.
    """
    model = joblib.load(model_path)
    with open(metrics_path, "r") as f:
        metrics_dict = json.load(f)
    return model, metrics_dict

