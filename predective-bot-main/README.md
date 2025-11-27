# Predicting Machine Failures with Classic ML  
_A Benchmark on the AI4I 2020 Predictive Maintenance Dataset_

This project is my technical assignment for the **ML/AI Intern** role at **Process Point Technologies (PPT)**.

The goal is to design and run a **small but realistic benchmark** for **predictive maintenance** on industrial-style data:

> Given operating conditions of a milling machine (temperatures, rotational speed, torque, tool wear, product type), predict whether the machine is in a **failure state**.

I compare three classic ML models:

1. **Logistic Regression** (baseline)
2. **Random Forest**
3. **XGBoost**

The focus is on handling **class imbalance** (machine failures are rare) and using the right metrics:  
**ROC AUC, Precision–Recall AUC, and F1-score for the failure class**, not just accuracy.

---

## 1. Repository Structure

```text
.
├─ data/
│  └─ ai4i2020.csv                   # AI4I 2020 Predictive Maintenance dataset
├─ article/
│  ├─ predictive_maintenance_ai4i_report.md
│  ├─ project_documentation_ai4i.md
│  ├─ class_balance.png
│  ├─ roc_curves.png
│  ├─ pr_curves.png
│  ├─ metrics_comparison.png
│  ├─ feature_importance_xgb.png
│  ├─ confusion_matrix_best.png
│  └─ torque_toolwear_scatter.png
└─ src/
   └─ ai4i_train_eval.py             # main experiment script
```

- `article/predictive_maintenance_ai4i_report.md`  
  → The **technical article** (in Markdown) with all figures embedded.

- `article/project_documentation_ai4i.md`  
  → A more formal **project documentation** report.

- `src/ai4i_train_eval.py`  
  → A single Python script that:
  - Loads and preprocesses the AI4I dataset
  - Trains Logistic Regression, Random Forest, and XGBoost
  - Computes relevant metrics
  - Generates all PNG figures used in the article

---

## 2. Dataset

I use the **AI4I 2020 Predictive Maintenance Dataset** (10,000 samples) which is a synthetic but realistic industrial dataset designed for predictive-maintenance tasks.

Key columns used:

- `type` – product quality / type (`L`, `M`, `H`)
- `air temperature [K]`
- `process temperature [K]`
- `rotational speed [rpm]`
- `torque [Nm]`
- `tool wear [min]`
- `machine failure` – 1 if any failure mode is active, else 0

I also engineer a simple feature:

- `temp_diff_k = process_temperature - air_temperature`

> Note: The script is robust to small variations in the column names; it uses substring matching in `load_ai4i()`.

---

## 3. Installation

### 3.1 Clone the repository

```bash
git clone <your-github-repo-url>.git
cd <your-repo-folder>
```

### 3.2 Create and activate a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
```

### 3.3 Install dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, you can use:

```text
pandas
scikit-learn
matplotlib
seaborn
xgboost
```

Install them via:

```bash
pip install pandas scikit-learn matplotlib seaborn xgboost
```

---

## 4. How to Reproduce the Experiments

### 4.1 Download the dataset

1. Download the `ai4i2020.csv` file from the AI4I 2020 Predictive Maintenance dataset (UCI / Kaggle).
2. Place it in the `data/` directory:

```text
data/ai4i2020.csv
```

### 4.2 Run the script

From the repo root:

```bash
python src/ai4i_train_eval.py
```

This will:

- Load `data/ai4i2020.csv`
- Clean and rename columns
- Engineer `temp_diff_k`
- One-hot encode `type`
- Split the data into 80% train / 20% test (stratified by `machine_failure`)
- Train:
  - Logistic Regression (with `StandardScaler` + `class_weight="balanced"`)
  - Random Forest (`n_estimators=300`, class-weighted)
  - XGBoost (`n_estimators=300`, max_depth=5, etc.)
- Print metrics to the console:
  - Accuracy
  - ROC AUC
  - Precision–Recall AUC
  - F1-score for the failure class
  - Full `classification_report` for each model
- Save all figures into `article/`:
  - `class_balance.png`
  - `roc_curves.png`
  - `pr_curves.png`
  - `metrics_comparison.png`
  - `feature_importance_xgb.png`
  - `confusion_matrix_best.png`
  - `torque_toolwear_scatter.png`

After this, opening `article/predictive_maintenance_ai4i_report.md` or `article/project_documentation_ai4i.md` in a Markdown viewer (or on GitHub) will show the full analysis and embedded charts.

---

## 5. Results Summary

On my 80/20 train–test split (stratified), the approximate metrics were:

| Model               | Accuracy | ROC AUC | PR AUC (failure) | F1 (failure) |
|---------------------|----------|---------|------------------|--------------|
| Logistic Regression | 0.975    | 0.973   | 0.61             | 0.72         |
| Random Forest       | 0.985    | 0.991   | 0.72             | 0.79         |
| XGBoost             | 0.988    | 0.994   | 0.77             | 0.83         |

Key points:

- All models have high accuracy because **most samples are normal** (no failure).
- Tree-based ensembles (**Random Forest, XGBoost**) significantly improve **failure recall**, **PR AUC**, and **F1-score** over the linear baseline.
- **XGBoost** provides the best trade-off between catching failures and avoiding excessive false alarms.

---

## 6. Visualizations

The report includes the following visualizations:

1. **Class balance bar chart** – `class_balance.png`  
2. **ROC curves** – `roc_curves.png`  
3. **Precision–Recall curves** – `pr_curves.png`  
4. **Metric comparison bar chart** – `metrics_comparison.png`  
5. **XGBoost feature importances** – `feature_importance_xgb.png`  
6. **Confusion matrix (best model)** – `confusion_matrix_best.png`  
7. **Torque vs Tool Wear scatter** – `torque_toolwear_scatter.png`  

---

## 7. Notes & Future Work

If I had more time, next steps would include:

- Moving from static tabular features to **time-series modeling** and remaining useful life (RUL).
- Trying more advanced imbalance handling (SMOTE, focal loss, cost-sensitive thresholds).
- Performing **hyperparameter tuning** (e.g., grid search / Optuna) for XGBoost.
- Calibrating probabilities (Platt scaling / isotonic regression) for better decision thresholds in production.
- Adding **explainability** (e.g., SHAP) for detailed per-sample explanations in industrial settings.
