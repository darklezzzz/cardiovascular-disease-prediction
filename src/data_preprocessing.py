"""
Data Preprocessing Module
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    Load cardiovascular disease dataset

    Parameters:
    -----------
    filepath : str
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    df = pd.read_csv(filepath, delimiter=";")
    print(f"Data loaded successfully!")
    print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def check_missing_values(df):
    """
    Check for missing values in dataset

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        Summary of missing values
    """
    missing_values = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    missing_df = pd.DataFrame(
        {
            "Feature": missing_values.index,
            "Missing Values": missing_values.values,
            "Percentage": missing_percentage.values,
        }
    )

    return missing_df


def create_features(df):
    """
    Create derived features

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        Dataframe with new features
    """
    df_copy = df.copy()

    # Convert age from days to years
    df_copy["age_years"] = df_copy["age"] / 365.25

    # Calculate BMI (Body Mass Index)
    df_copy["bmi"] = df_copy["weight"] / ((df_copy["height"] / 100) ** 2)

    return df_copy


def remove_outliers(df):
    """
    Remove outliers based on medical knowledge

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe

    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    tuple
        (removed_count, removed_percentage)
    """
    initial_count = len(df)

    # Filter by realistic medical values
    df_clean = df[
        (df["height"] >= 140)
        & (df["height"] <= 210)
        & (df["weight"] >= 40)
        & (df["weight"] <= 200)
        & (df["ap_hi"] >= 80)
        & (df["ap_hi"] <= 250)
        & (df["ap_lo"] >= 50)
        & (df["ap_lo"] <= 150)
        & (df["ap_hi"] > df["ap_lo"])
        & (df["bmi"] >= 15)
        & (df["bmi"] <= 60)
    ]

    removed_count = initial_count - len(df_clean)
    removed_percentage = (removed_count / initial_count) * 100

    print(f"Removed outliers: {removed_count} ({removed_percentage:.2f}%)")
    print(f"Remaining records: {len(df_clean)}")

    return df_clean, (removed_count, removed_percentage)


def prepare_features(df, target_column="cardio"):
    """
    Prepare features for modeling

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of target column

    Returns:
    --------
    tuple
        (X, y) - Features and target
    """
    features_to_use = [
        "age_years",
        "gender",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "bmi",
    ]

    X = df[features_to_use]
    y = df[target_column]

    return X, y


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test and apply standardization

    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    test_size : float
        Proportion of test set
    random_state : int
        Random seed

    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index
    )

    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_feature_descriptions():
    """
    Get dictionary of feature descriptions

    Returns:
    --------
    dict
        Feature names and their descriptions
    """
    return {
        "age": "Age (in days)",
        "gender": "Gender (1 - female, 2 - male)",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "ap_hi": "Systolic blood pressure",
        "ap_lo": "Diastolic blood pressure",
        "cholesterol": "Cholesterol (1: normal, 2: above normal, 3: well above normal)",
        "gluc": "Glucose (1: normal, 2: above normal, 3: well above normal)",
        "smoke": "Smoking (0: no, 1: yes)",
        "alco": "Alcohol intake (0: no, 1: yes)",
        "active": "Physical activity (0: no, 1: yes)",
        "cardio": "Presence of cardiovascular disease (target variable)",
        "age_years": "Age in years (derived)",
        "bmi": "Body Mass Index (derived)",
    }
