"""
Utility Functions
Helper functions for the project
"""

import os
import json
import pickle


def ensure_dir_exists(directory):
    """
    Create directory if it doesn't exist

    Parameters:
    -----------
    directory : str
        Directory path
    """
    os.makedirs(directory, exist_ok=True)


def save_json(data, filepath):
    """
    Save dictionary to JSON file

    Parameters:
    -----------
    data : dict
        Data to save
    filepath : str
        Path to save file
    """
    ensure_dir_exists(os.path.dirname(filepath))

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved to {filepath}")


def load_json(filepath):
    """
    Load JSON file

    Parameters:
    -----------
    filepath : str
        Path to JSON file

    Returns:
    --------
    dict
        Loaded data
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    return data


def print_section_header(title, width=80):
    """
    Print formatted section header

    Parameters:
    -----------
    title : str
        Section title
    width : int
        Width of the header
    """
    print("\n" + "=" * width)
    padding = (width - len(title)) // 2
    print(" " * padding + title)
    print("=" * width + "\n")


def format_metrics(metrics_dict, decimals=4):
    """
    Format metrics dictionary for display

    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with metrics
    decimals : int
        Number of decimal places

    Returns:
    --------
    dict
        Formatted metrics
    """
    formatted = {}
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            formatted[key] = round(value, decimals)
        else:
            formatted[key] = value

    return formatted


def get_project_root():
    """
    Get project root directory

    Returns:
    --------
    str
        Project root path
    """
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(src_dir)

    return project_root


def print_model_summary(model, model_name):
    """
    Print model summary

    Parameters:
    -----------
    model : object
        Trained model
    model_name : str
        Name of the model
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Summary")
    print(f"{'='*60}")
    print(f"Model Type: {type(model).__name__}")

    if hasattr(model, "get_params"):
        print("\nKey Parameters:")
        params = model.get_params()
        for key, value in list(params.items())[:5]:
            print(f"  {key}: {value}")

    print(f"{'='*60}\n")
