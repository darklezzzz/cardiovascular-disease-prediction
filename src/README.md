# Source Code Directory

This directory contains the modular Python implementation of the cardiovascular disease prediction pipeline.

## Modules

### 📦 `data_preprocessing.py`
Data loading, cleaning, and feature engineering functions:
- `load_data()` - Load dataset from CSV
- `check_missing_values()` - Check for missing data
- `create_features()` - Create derived features (age_years, BMI)
- `remove_outliers()` - Remove medical outliers
- `prepare_features()` - Prepare X and y for modeling
- `split_and_scale_data()` - Train/test split and standardization

### 🤖 `model_training.py`
Model training and persistence:
- `train_logistic_regression()` - Train LR model
- `train_random_forest()` - Train RF model
- `train_xgboost()` - Train XGBoost model
- `train_all_models()` - Train all three models
- `save_model()` / `load_model()` - Model persistence

### 📊 `model_evaluation.py`
Model evaluation and metrics:
- `evaluate_model()` - Calculate all metrics
- `evaluate_all_models()` - Evaluate and compare models
- `calculate_confusion_matrix()` - Confusion matrix
- `calculate_roc_curve()` - ROC curve data
- `get_feature_importance()` - Extract feature importance
- `print_evaluation_summary()` - Formatted results

### 📈 `visualization.py`
Plotting and visualization:
- `plot_target_distribution()` - Target variable plots
- `plot_correlation_matrix()` - Correlation heatmap
- `plot_model_comparison()` - Metrics comparison
- `plot_confusion_matrices()` - Confusion matrices
- `plot_roc_curves()` - ROC curves
- `plot_feature_importance()` - Feature importance

### 🛠️ `utils.py`
Utility functions:
- `ensure_dir_exists()` - Directory creation
- `save_json()` / `load_json()` - JSON operations
- `print_section_header()` - Formatted headers
- `format_metrics()` - Format metric values
- `get_project_root()` - Get project path

## Usage

### As a Package
```python
from src.data_preprocessing import load_data, create_features
from src.model_training import train_xgboost
from src.model_evaluation import evaluate_model

# Load and preprocess data
df = load_data('data/cardio_train.csv')
df = create_features(df)

# Train model
model = train_xgboost(X_train, y_train)

# Evaluate
metrics = evaluate_model(y_test, y_pred, y_pred_proba, 'XGBoost')
```

### Run Complete Pipeline
```bash
python main.py
```

## Dependencies

All modules require packages listed in `requirements.txt`:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- shap

## Design Principles

1. **Modularity**: Each module has a single responsibility
2. **Reusability**: Functions can be imported and used independently
3. **Documentation**: All functions have docstrings
4. **Type Hints**: Clear parameter and return types
5. **Error Handling**: Graceful handling of common errors

## Testing

To test individual modules:

```python
# Test data preprocessing
from src.data_preprocessing import load_data, create_features
df = load_data('data/cardio_train.csv')
df = create_features(df)
print(df.head())

# Test model training
from src.model_training import train_logistic_regression
model = train_logistic_regression(X_train, y_train)
print("Model trained successfully!")
```

## Extending the Project

To add new models:
1. Add training function to `model_training.py`
2. Update `train_all_models()` to include new model
3. Evaluation will automatically work with new model

To add new visualizations:
1. Add plotting function to `visualization.py`
2. Follow existing patterns (save_path parameter, etc.)
3. Call from main.py or notebook
