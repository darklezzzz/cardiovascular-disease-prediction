# 🫀 Cardiovascular Disease Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Lines](https://img.shields.io/badge/Code-1139%20lines-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.799-success.svg)]()

> **A comprehensive machine learning project for cardiovascular disease prediction with modular Python architecture, achieving 73.2% accuracy and 0.799 ROC-AUC score.**

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Results](#-key-results)
- [Dataset](#-dataset)
- [Workflow](#-workflow)
- [Model Performance](#-model-performance)
- [Top Risk Factors](#-top-risk-factors)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Architecture & Design](#️-architecture--design)
- [Tech Stack](#-tech-stack)
- [Insights](#-insights)
- [Model Explainability](#-model-explainability)
- [Future Improvements](#-future-improvements)
- [Usage Examples](#-usage-examples)
- [Inspiration](#-inspiration)
- [Academic Context](#-academic-context)
- [Contact](#-contact)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## �📋 Overview
A machine learning project for **predicting cardiovascular disease risk** using physiological and behavioral patient data.  
Three classification models were trained and compared — achieving a **ROC-AUC of 0.799** on the test set.

### 🎯 Project Highlights
- 🏗️ **Modular Architecture:** 1,139 lines of clean, reusable Python code
- 🤖 **3 ML Models:** Logistic Regression, Random Forest, XGBoost
- 📊 **Comprehensive Analysis:** 45-cell Jupyter notebook with full pipeline
- 📈 **10 Visualizations:** High-quality plots for presentation and analysis
- 🔍 **Explainable AI:** SHAP values for model interpretation
- 📚 **Well-Documented:** Complete documentation and usage examples

---

## 🎯 Key Results
- **Best Model:** XGBoost  
- **Accuracy:** 73.2%  
- **ROC-AUC:** 0.799 (strong performance for medical diagnostics)  
- **Primary Risk Factor:** Systolic blood pressure (48% importance)  
- **Dataset:** 68,426 records after cleaning  

---

## 📊 Dataset
[Kaggle – Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

- 70,000 patient records  
- 11 features + target variable  
- Balanced classes (~50/50)

---

## 🔬 Workflow
1. **Exploratory Data Analysis (EDA)**
   - Distribution and correlation analysis  
   - Outlier removal (~2.2%)  

2. **Preprocessing**
   - Feature engineering (BMI, age in years)  
   - Standard scaling and 80/20 train-test split  

3. **Model Training**
   - Logistic Regression (baseline)  
   - Random Forest  
   - XGBoost (best performer)  

4. **Evaluation & Explainability**
   - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
   - Feature Importance and SHAP analysis  

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:------|:---------:|:----------:|:-------:|:---------:|:--------:|
| Logistic Regression | 72.7% | 75.2% | 66.9% | 70.8% | 0.791 |
| Random Forest | 73.2% | 74.8% | 69.3% | 71.9% | 0.798 |
| **XGBoost** | **73.2%** | **74.7%** | **69.4%** | **72.0%** | **0.799** |

---

## 🔑 Top Risk Factors
1. Systolic Blood Pressure — 48.1%  
2. Cholesterol — 14.5%  
3. Diastolic Pressure — 12.6%  
4. Age — 5.5%  
5. Glucose — 3.4%  

---

## 📁 Project Structure
```
cardiovascular-disease-prediction/
│
├── 📂 src/                          # Python source code (1,040 lines)
│   ├── __init__.py                  # Package initialization (7 lines)
│   ├── data_preprocessing.py        # Data loading and cleaning (232 lines)
│   ├── model_training.py            # ML model training (198 lines)
│   ├── model_evaluation.py          # Metrics and evaluation (194 lines)
│   ├── visualization.py             # Plotting functions (266 lines)
│   ├── utils.py                     # Helper utilities (143 lines)
│   └── README.md                    # Module documentation
│
├── 📓 notebooks/                    # Jupyter notebooks
│   └── cardiovascular_analysis.ipynb  # Full analysis pipeline (45 cells)
│
├── 📊 results/                      # Visualizations (10 PNG files)
│   ├── confusion_matrices.png       # Confusion matrices (3 models)
│   ├── correlation_matrix.png       # Feature correlations
│   ├── feature_importance.png       # Feature importance (RF & XGBoost)
│   ├── models_comparison.png        # Model metrics comparison
│   ├── outliers_boxplots.png        # Outlier detection
│   ├── risk_factors_analysis.png    # Risk factors distribution
│   ├── roc_curves.png               # ROC curves comparison
│   ├── shap_importance.png          # SHAP feature importance
│   ├── shap_summary.png             # SHAP summary plot
│   └── target_distribution.png      # Class balance
│
├── 🤖 models/                       # Trained models
│   ├── logistic_regression_model.pkl  # Logistic Regression (1 KB)
│   ├── random_forest_model.pkl      # Random Forest (27 MB)
│   ├── xgboost_model.pkl            # XGBoost (411 KB)
│   └── README.md                    # Model documentation
│
├── 📁 data/                         # Dataset
│   ├── cardio_train.csv             # 70k patient records
│   └── README.md                    # Data documentation
│
├── � .vscode/                      # VS Code configuration
│   └── settings.json                # Python analysis settings
│
├── 📝 Configuration Files
│   ├── .gitignore                   # Git ignore rules
│   ├── .gitattributes               # GitHub language detection
│   ├── requirements.txt             # Python dependencies
│   └── LICENSE                      # MIT license
│
├── 📄 README.md                     # This file (project documentation)
├── 🐍 main.py                       # Main pipeline script (99 lines)
└── 🐍 venv/                         # Virtual environment (excluded from Git)
```

---

## 🚀 Quick Start

### Option 1: Run Python Pipeline (Manual)
```bash
# 1. Clone repository
git clone https://github.com/darklezzzz/cardiovascular-disease-prediction.git
cd cardiovascular-disease-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset from Kaggle and place in data/

# 5. Run complete pipeline
python main.py
```

### Option 2: Interactive Jupyter Notebook
```bash
# 1-4. Same as Option 1 above

# 5. Launch Jupyter
jupyter notebook notebooks/cardiovascular_analysis.ipynb

# 6. Run all cells
```

### Option 3: Use as Python Package
```python
# Import modules and run custom workflow
from src.data_preprocessing import load_data, create_features, remove_outliers, prepare_features, split_and_scale_data
from src.model_training import train_xgboost
from src.model_evaluation import evaluate_model

# Load and preprocess data
df = load_data('data/cardio_train.csv')
df = create_features(df)
df_clean, _ = remove_outliers(df)

# Prepare features
X, y = prepare_features(df_clean)
X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)

# Train model
model = train_xgboost(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate
metrics = evaluate_model(y_test, y_pred, y_pred_proba, 'XGBoost')
print(metrics)
```

---

## 🏗️ Architecture & Design

### Modular Structure
The project follows a **modular architecture** with clear separation of concerns:

```
Data Layer          →  data_preprocessing.py
Model Layer         →  model_training.py
Evaluation Layer    →  model_evaluation.py
Visualization Layer →  visualization.py
Utilities           →  utils.py
```

### Key Features
✅ **Reusable Components** — Functions can be imported and used independently  
✅ **Clean Code** — Well-documented with docstrings and type hints  
✅ **Scalable** — Easy to add new models or features  
✅ **Professional** — Follows Python best practices  
✅ **Dual Interface** — Use via CLI (`main.py`) or Jupyter notebooks  

### Code Statistics
- **Total Python Code:** 1,139 lines
- **Modules:** 6 Python files
- **Functions:** 40+ reusable functions
- **Documentation:** 100% docstring coverage

---

## 🧰 Tech Stack

### Core Technologies
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-critical?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

### Python Modules Overview

| Module | Purpose | Lines | Key Functions |
|--------|---------|-------|---------------|
| `data_preprocessing.py` | Data loading & cleaning | 232 | `load_data()`, `remove_outliers()`, `create_features()` |
| `model_training.py` | ML model training | 198 | `train_xgboost()`, `train_random_forest()`, `save_model()` |
| `model_evaluation.py` | Performance metrics | 194 | `evaluate_model()`, `calculate_roc_curve()` |
| `visualization.py` | Plotting & charts | 266 | `plot_roc_curves()`, `plot_model_comparison()` |
| `utils.py` | Helper functions | 143 | `save_json()`, `print_section_header()` |
| `main.py` | Pipeline execution | 99 | Complete workflow orchestration |

---

## 💡 Insights
✅ ROC-AUC of 0.799 shows robust discrimination ability  
✅ Identifies ~69% of patients with cardiovascular disease (Recall)  
✅ Systolic pressure — most influential variable  
✅ Model suitable for preliminary clinical screening  

---

## 🧠 Model Explainability
- **Feature Importance:** Highlights global predictors  
- **SHAP Summary:** Explains individual predictions  
- **Confusion Matrices:** Shows model misclassifications  
- **ROC Curves:** Compares classifier performance  

---

## 🚧 Future Improvements

### Data & Features
- [ ] Include extended medical history and genetic markers
- [ ] Add time-series patient data for longitudinal analysis
- [ ] Integrate additional clinical biomarkers

### Models & Algorithms
- [ ] Implement neural networks (MLP, CNN for medical imaging)
- [ ] Apply advanced ensembling techniques (stacking, blending)
- [ ] Experiment with CatBoost, LightGBM
- [ ] Hyperparameter optimization with Optuna/Ray Tune

### Explainability
- [ ] Add LIME for local interpretability
- [ ] Individual patient risk reports
- [ ] Counterfactual explanations

### Deployment
- [ ] Web interface using Streamlit or Gradio
- [ ] REST API with FastAPI
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] CI/CD pipeline with GitHub Actions

### Testing & Validation
- [ ] Unit tests for all modules (pytest)
- [ ] Cross-validation and bootstrap validation
- [ ] External validation on different datasets  

---

## 📚 Usage Examples

### Example 1: Train Single Model
```python
from src.data_preprocessing import load_data, create_features, split_and_scale_data
from src.model_training import train_xgboost
from src.model_evaluation import evaluate_model

# Load and prepare data
df = load_data('data/cardio_train.csv')
df = create_features(df)
X, y = prepare_features(df)
X_train, X_test, y_train, y_test, _ = split_and_scale_data(X, y)

# Train XGBoost
model = train_xgboost(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
metrics = evaluate_model(y_test, y_pred, y_pred_proba, 'XGBoost')
print(metrics)
```

### Example 2: Compare All Models
```python
from src.model_training import train_all_models
from src.model_evaluation import evaluate_all_models

# Train all three models
models = train_all_models(X_train, y_train, save_dir='models')

# Compare performance
results_df, predictions = evaluate_all_models(models, X_test, y_test)
print(results_df)
```

### Example 3: Generate Visualizations
```python
from src.visualization import plot_roc_curves, plot_model_comparison

# Plot ROC curves
plot_roc_curves(y_test, predictions, save_path='results/roc_curves.png')

# Plot metrics comparison
plot_model_comparison(results_df, save_path='results/models_comparison.png')
```

### Example 4: Custom Pipeline
```python
# Build custom workflow
from src import data_preprocessing as dp
from src import model_training as mt

# Custom preprocessing
df = dp.load_data('data/cardio_train.csv')
df = dp.create_features(df)
df_clean, stats = dp.remove_outliers(df)

# Train with custom parameters
custom_model = mt.train_xgboost(
    X_train, y_train,
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05
)
```

---

## 🎓 Inspiration

This project is part of my academic preparation for pursuing a **Master’s degree in Artificial Intelligence in France**.  
My goal is to build a solid portfolio that reflects both **technical competence** and **analytical depth** in the field of **machine learning, data science, and AI systems**.

Through projects like this one, I aim to demonstrate my ability to design, train, and interpret intelligent models while maintaining a strong focus on clarity, reproducibility, and real-world applicability.  

This repository represents not only a learning milestone but also a step toward contributing to the international AI research community — combining rigorous engineering with the ethical and human-centered vision that defines modern French academia.

---

## 👨🏻‍🏫 Academic Context

### Skills Demonstrated
This project showcases comprehensive data science and software engineering capabilities:

#### Machine Learning
✅ End-to-end ML pipeline development  
✅ Multiple model comparison (Logistic Regression, Random Forest, XGBoost)  
✅ Proper validation methodology (train-test split, stratification)  
✅ Advanced metrics (ROC-AUC, precision, recall, F1-score)  
✅ Model interpretation (SHAP, feature importance)

#### Software Engineering
✅ Modular code architecture with clear separation of concerns  
✅ Reusable Python packages with comprehensive documentation  
✅ Version control with Git and GitHub  
✅ Professional project structure  
✅ Clean code principles and best practices

#### Domain Knowledge
✅ Healthcare/medical data analysis  
✅ Feature engineering based on domain expertise (BMI, outlier removal)  
✅ Clinical relevance of predictions (recall vs precision trade-offs)  
✅ Interpretability for medical decision-making

#### Data Science Workflow
✅ Exploratory Data Analysis (EDA)  
✅ Data cleaning and preprocessing  
✅ Feature engineering  
✅ Statistical validation  
✅ Result visualization and communication

---

## 📬 Contact

**Author:** Ivanov Artyom  
**GitHub:** [@darklezzzz](https://github.com/darklezzzz)  
**Email:** darklezzzz@hotmail.com  
**Repository:** [cardiovascular-disease-prediction](https://github.com/darklezzzz/cardiovascular-disease-prediction)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset:** [Kaggle - Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Libraries:** scikit-learn, XGBoost, SHAP, pandas, matplotlib
- **Inspiration:** Real-world healthcare challenges in cardiovascular disease prevention

---

<div align="center">

### ⭐ If this project helped you, consider giving it a star! ⭐

</div>
