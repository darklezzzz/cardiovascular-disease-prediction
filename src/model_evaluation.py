"""
Model Evaluation Module
Handles model evaluation and metrics calculation
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """
    Calculate all evaluation metrics for a model

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model

    Returns:
    --------
    dict
        Dictionary with all metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
    }


def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all models and return comparison dataframe

    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : array-like
        Test features
    y_test : array-like
        Test target

    Returns:
    --------
    pd.DataFrame
        Comparison of all models
    """
    results = []
    predictions = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        predictions[name] = {"y_pred": y_pred, "y_pred_proba": y_pred_proba}

        metrics = evaluate_model(y_test, y_pred, y_pred_proba, name)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("Model")

    return results_df, predictions


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels

    Returns:
    --------
    np.ndarray
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def calculate_roc_curve(y_true, y_pred_proba):
    """
    Calculate ROC curve data

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities

    Returns:
    --------
    tuple
        (fpr, tpr, thresholds)
    """
    return roc_curve(y_true, y_pred_proba)


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from tree-based models

    Parameters:
    -----------
    model : object
        Trained model (RandomForest or XGBoost)
    feature_names : list
        List of feature names

    Returns:
    --------
    pd.DataFrame
        Feature importance sorted by value
    """
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return importance_df


def print_evaluation_summary(results_df):
    """
    Print formatted evaluation summary

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe from evaluate_all_models
    """
    print("=" * 90)
    print(" " * 28 + "MODEL COMPARISON")
    print("=" * 90)
    print(results_df.round(4))
    print("=" * 90)

    # Find best model
    best_model = results_df["ROC-AUC"].idxmax()
    best_auc = results_df.loc[best_model, "ROC-AUC"]

    print(f"\n🎯 Best Model: {best_model}")
    print(f"   ROC-AUC: {best_auc:.4f}")
    print("=" * 90)


def calculate_classification_report(y_true, y_pred):
    """
    Generate detailed classification report

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels

    Returns:
    --------
    dict
        Classification report as dictionary
    """
    from sklearn.metrics import classification_report

    return classification_report(y_true, y_pred, output_dict=True)
