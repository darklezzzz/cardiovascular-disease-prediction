"""
Model Training Module
Handles training of classification models
"""

import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_logistic_regression(X_train, y_train, **kwargs):
    """
    Train Logistic Regression model

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    **kwargs : dict
        Additional parameters for LogisticRegression

    Returns:
    --------
    LogisticRegression
        Trained model
    """
    print("Training Logistic Regression model...")

    params = {"max_iter": 1000, "random_state": 42}
    params.update(kwargs)

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    print("Logistic Regression model trained!")
    return model


def train_random_forest(X_train, y_train, **kwargs):
    """
    Train Random Forest model

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    **kwargs : dict
        Additional parameters for RandomForestClassifier

    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    print("Training Random Forest model...")

    params = {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(kwargs)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    print("Random Forest model trained!")
    return model


def train_xgboost(X_train, y_train, **kwargs):
    """
    Train XGBoost model

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    **kwargs : dict
        Additional parameters for XGBClassifier

    Returns:
    --------
    XGBClassifier
        Trained model
    """
    print("Training XGBoost model...")

    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
    }
    params.update(kwargs)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    print("XGBoost model trained!")
    return model


def save_model(model, filepath):
    """
    Save trained model to file

    Parameters:
    -----------
    model : object
        Trained model
    filepath : str
        Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from file

    Parameters:
    -----------
    filepath : str
        Path to the model file

    Returns:
    --------
    object
        Loaded model
    """
    with open(filepath, "rb") as f:
        model = pickle.load(f)

    print(f"Model loaded from {filepath}")
    return model


def train_all_models(X_train, y_train, save_dir="models"):
    """
    Train all three models

    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    save_dir : str
        Directory to save models

    Returns:
    --------
    dict
        Dictionary with trained models
    """
    import os

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    models = {}

    # Train Logistic Regression
    models["logistic_regression"] = train_logistic_regression(X_train, y_train)
    save_model(
        models["logistic_regression"], f"{save_dir}/logistic_regression_model.pkl"
    )

    # Train Random Forest
    models["random_forest"] = train_random_forest(X_train, y_train)
    save_model(models["random_forest"], f"{save_dir}/random_forest_model.pkl")

    # Train XGBoost
    models["xgboost"] = train_xgboost(X_train, y_train)
    save_model(models["xgboost"], f"{save_dir}/xgboost_model.pkl")

    print("\n All models trained and saved!")
    return models
