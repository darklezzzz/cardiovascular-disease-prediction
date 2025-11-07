#!/usr/bin/env python3
"""
Main script to run the complete ML pipeline
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_preprocessing import (
    load_data,
    create_features,
    remove_outliers,
    prepare_features,
    split_and_scale_data,
)
from model_training import train_all_models
from model_evaluation import evaluate_all_models, print_evaluation_summary
from visualization import (
    plot_target_distribution,
    plot_correlation_matrix,
    plot_model_comparison,
    plot_roc_curves,
)
from utils import print_section_header


def main():
    """Run complete ML pipeline"""

    print_section_header("CARDIOVASCULAR DISEASE PREDICTION PIPELINE")

    # 1. Load Data
    print_section_header("1. LOADING DATA")
    data_path = PROJECT_ROOT / "data" / "cardio_train.csv"

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please download the dataset from Kaggle and place it in data/")
        return

    df = load_data(str(data_path))

    # 2. Preprocessing
    print_section_header("2. DATA PREPROCESSING")
    df = create_features(df)
    df_clean, (removed, pct) = remove_outliers(df)

    # 3. Prepare Features
    print_section_header("3. FEATURE PREPARATION")
    X, y = prepare_features(df_clean)
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)

    # 4. Train Models
    print_section_header("4. MODEL TRAINING")
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    models = train_all_models(X_train, y_train, save_dir=str(models_dir))

    # 5. Evaluate Models
    print_section_header("5. MODEL EVALUATION")
    results_df, predictions = evaluate_all_models(models, X_test, y_test)
    print_evaluation_summary(results_df)

    # 6. Visualizations
    print_section_header("6. GENERATING VISUALIZATIONS")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print("Plotting target distribution...")
    plot_target_distribution(
        df_clean, save_path=str(results_dir / "target_distribution.png")
    )

    print("Plotting correlation matrix...")
    plot_correlation_matrix(
        df_clean, save_path=str(results_dir / "correlation_matrix.png")
    )

    print("Plotting model comparison...")
    plot_model_comparison(
        results_df, save_path=str(results_dir / "models_comparison.png")
    )

    print("Plotting ROC curves...")
    plot_roc_curves(y_test, predictions, save_path=str(results_dir / "roc_curves.png"))

    print_section_header("PIPELINE COMPLETED SUCCESSFULLY!")

    return results_df, models


if __name__ == "__main__":
    results, models = main()
