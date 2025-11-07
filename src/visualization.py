"""
Visualization Module
Handles all plotting and visualization functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_target_distribution(df, target_column="cardio", save_path=None):
    """
    Plot target variable distribution

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Name of target column
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cardio_counts = df[target_column].value_counts()

    # Bar chart
    axes[0].bar(
        ["No Disease", "Has Disease"],
        cardio_counts.values,
        color=["#2ecc71", "#e74c3c"],
        alpha=0.7,
        edgecolor="black",
    )
    axes[0].set_ylabel("Number of Patients", fontsize=12)
    axes[0].set_title("Target Variable Distribution", fontsize=14, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Add percentages
    for i, v in enumerate(cardio_counts.values):
        percentage = (v / len(df)) * 100
        axes[0].text(
            i,
            v + 500,
            f"{v}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Pie chart
    axes[1].pie(
        cardio_counts.values,
        labels=["No Disease", "Has Disease"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ecc71", "#e74c3c"],
        explode=(0, 0.05),
        shadow=True,
    )
    axes[1].set_title("Target Variable Proportions", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_correlation_matrix(df, save_path=None):
    """
    Plot correlation matrix heatmap

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    save_path : str, optional
        Path to save the figure
    """
    correlation_matrix = df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        linewidths=1,
        cbar_kws={"label": "Correlation Coefficient"},
    )
    plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_model_comparison(results_df, save_path=None):
    """
    Plot comparison of model metrics

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe with metrics
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    results_df.plot(kind="bar", ax=ax, width=0.8, edgecolor="black")
    ax.set_title(
        "Classification Model Metrics Comparison",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=12, fontweight="bold")
    ax.set_xticklabels(results_df.index, rotation=45, ha="right")
    ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1])

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_confusion_matrices(y_test, predictions, save_path=None):
    """
    Plot confusion matrices for all models

    Parameters:
    -----------
    y_test : array-like
        True labels
    predictions : dict
        Dictionary with predictions for each model
    save_path : str, optional
        Path to save the figure
    """
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    model_names = list(predictions.keys())

    for idx, (model_name, preds) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, preds["y_pred"])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            cbar_kws={"label": "Count"},
        )
        axes[idx].set_title(
            f"{model_name}\nConfusion Matrix", fontsize=14, fontweight="bold"
        )
        axes[idx].set_xlabel("Predicted Class", fontsize=11)
        axes[idx].set_ylabel("True Class", fontsize=11)
        axes[idx].set_xticklabels(["No", "Yes"])
        axes[idx].set_yticklabels(["No", "Yes"])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_roc_curves(y_test, predictions, save_path=None):
    """
    Plot ROC curves for all models

    Parameters:
    -----------
    y_test : array-like
        True labels
    predictions : dict
        Dictionary with predictions for each model
    save_path : str, optional
        Path to save the figure
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    plt.figure(figsize=(12, 8))

    colors = ["blue", "green", "red"]

    for idx, (model_name, preds) in enumerate(predictions.items()):
        fpr, tpr, _ = roc_curve(y_test, preds["y_pred_proba"])
        roc_auc = roc_auc_score(y_test, preds["y_pred_proba"])
        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {roc_auc:.4f})",
            linewidth=2.5,
            color=colors[idx],
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random Classifier (AUC = 0.5)")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=13, fontweight="bold")
    plt.ylabel("True Positive Rate (TPR)", fontsize=13, fontweight="bold")
    plt.title("ROC Curves for All Models", fontsize=16, fontweight="bold", pad=20)
    plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_feature_importance(importance_df, model_name, save_path=None):
    """
    Plot feature importance

    Parameters:
    -----------
    importance_df : pd.DataFrame
        Feature importance dataframe
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.barh(
        range(len(importance_df)),
        importance_df["importance"],
        color="steelblue",
        alpha=0.7,
        edgecolor="black",
    )
    plt.yticks(range(len(importance_df)), importance_df["feature"])
    plt.xlabel("Feature Importance", fontsize=12, fontweight="bold")
    plt.title(f"Feature Importance - {model_name}", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
