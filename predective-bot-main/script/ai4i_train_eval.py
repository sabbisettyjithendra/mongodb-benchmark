"""Train and evaluate classic ML models on the AI4I 2020 Predictive Maintenance dataset.

This script:
- Loads data from data/ai4i2020.csv
- Does basic preprocessing & feature engineering
- Trains Logistic Regression, Random Forest, and XGBoost
- Computes metrics (Accuracy, ROC AUC, PR AUC, F1 for failure class)
- Generates all the PNG plots referenced in the markdown report:
    * class_balance.png
    * roc_curves.png
    * pr_curves.png
    * feature_importance_xgb.png
    * confusion_matrix_best.png
    * metrics_comparison.png
    * torque_toolwear_scatter.png

Expected repo layout:

    .
    ├─ data/
    │   └─ ai4i2020.csv
    ├─ article/
    │   └─ predictive_maintenance_ai4i_report.md
    └─ src/
        └─ ai4i_train_eval.py  (this file)

Run from the repo root:

    python src/ai4i_train_eval.py

The script will write all PNGs into the article/ folder so the markdown
can reference them directly.
"""  # noqa: E501

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit(
        "xgboost is required for this script. Install it with:\n"
        "  pip install xgboost"
    ) from e



OUTPUT_DIR = "article"


def load_ai4i(path: str) -> pd.DataFrame:
    """Load the AI4I 2020 dataset and return a cleaned DataFrame.

    The function is robust to slight variations in column naming by matching
    on substrings instead of exact names.
    """  # noqa: D401
    df = pd.read_csv("ai4i2020.csv")

    # Normalize column names to lower for easier matching (keep original as backup)
    orig_cols = list(df.columns)
    lower_cols = [c.strip().lower() for c in df.columns]
    df.columns = lower_cols

    # Map from detected column -> canonical name
    rename_map: Dict[str, str] = {}

    for c in df.columns:
        cl = c.lower()
        if "air temperature" in cl:
            rename_map[c] = "air_temp_k"
        elif "process temperature" in cl:
            rename_map[c] = "process_temp_k"
        elif "rotational speed" in cl:
            rename_map[c] = "rot_speed_rpm"
        elif "torque" in cl:
            rename_map[c] = "torque_nm"
        elif "tool wear" in cl:
            rename_map[c] = "tool_wear_min"
        elif "machine failure" in cl:
            rename_map[c] = "machine_failure"
        elif cl == "type":
            rename_map[c] = "type"

    df = df.rename(columns=rename_map)

    expected_cols = [
        "type",
        "air_temp_k",
        "process_temp_k",
        "rot_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "machine_failure",
    ]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}.\n"
            "Check your ai4i2020.csv column names and adjust the mapping in load_ai4i()."  # noqa: E501
        )

    df = df[expected_cols].copy()
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]

    return df


def make_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_class_balance(y: pd.Series, out_path: str) -> None:
    counts = y.value_counts().sort_index()
    labels = ["Normal (0)", "Failure (1)"]

    plt.figure()
    counts.plot(kind="bar")
    plt.xticks([0, 1], labels, rotation=0)
    plt.ylabel("Count")
    plt.title("Class Balance – Machine Failure")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> List[Dict]:
    """Train three models and return their results (metrics + scores)."""
    models: List[Tuple[str, object]] = []

    log_reg = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    models.append(("Logistic Regression", log_reg))

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    models.append(("Random Forest", rf))

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    models.append(("XGBoost", xgb))

    results: List[Dict] = []

    for name, model in models:
        print("\n=== Training", name, "===")
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict(X_test)

        y_pred = model.predict(X_test)

        acc = float((y_pred == y_test).mean())
        roc = float(roc_auc_score(y_test, y_score))
        pr_auc = float(average_precision_score(y_test, y_score))
        f1_fail = float(f1_score(y_test, y_pred, pos_label=1))

        print(
            f"{name}: acc={acc:.3f}, roc_auc={roc:.3f}, pr_auc={pr_auc:.3f}, f1_fail={f1_fail:.3f}"  # noqa: E501
        )
        print(classification_report(y_test, y_pred, digits=3))

        results.append(
            {
                "name": name,
                "model": model,
                "accuracy": acc,
                "roc_auc": roc,
                "pr_auc": pr_auc,
                "f1_fail": f1_fail,
                "y_pred": y_pred,
                "y_score": y_score,
            }
        )

    return results


def plot_roc_curves(results: List[Dict], y_test: pd.Series, out_path: str) -> None:
    plt.figure()
    for res in results:
        fpr, tpr, _ = roc_curve(y_test, res["y_score"])
        plt.plot(fpr, tpr, label=f"{res['name']} (AUC = {res['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves – Machine Failure Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pr_curves(results: List[Dict], y_test: pd.Series, out_path: str) -> None:
    plt.figure()
    for res in results:
        precision, recall, _ = precision_recall_curve(y_test, res["y_score"])
        plt.plot(recall, precision, label=f"{res['name']} (AP = {res['pr_auc']:.3f})")
    plt.xlabel("Recall (failure)")
    plt.ylabel("Precision (failure)")
    plt.title("Precision–Recall Curves – Machine Failure (Positive Class)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metrics_comparison(results: List[Dict], out_path: str) -> None:
    names = [r["name"] for r in results]
    f1_scores = [r["f1_fail"] for r in results]
    pr_aucs = [r["pr_auc"] for r in results]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, f1_scores, width, label="F1 (failure)")
    plt.bar(x + width / 2, pr_aucs, width, label="PR AUC (failure)")
    plt.xticks(x, names, rotation=15)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Comparison of F1 and PR AUC by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance_xgb(
    xgb_model: XGBClassifier, feature_names: List[str], out_path: str
) -> None:
    importances = xgb_model.feature_importances_
    idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(6, 4))
    plt.barh(
        np.array(feature_names)[idx][:10][::-1],
        importances[idx][:10][::-1],
    )
    plt.xlabel("Feature Importance (gain)")
    plt.title("XGBoost – Top Feature Importances")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(
    y_test: pd.Series, y_pred: np.ndarray, out_path: str
) -> None:
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal (0)", "Failure (1)"])
    ax.set_yticklabels(["Normal (0)", "Failure (1)"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix – Best Model")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_torque_toolwear_scatter(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    normal = df[df["machine_failure"] == 0]
    fail = df[df["machine_failure"] == 1]

    plt.scatter(
        normal["tool_wear_min"],
        normal["torque_nm"],
        alpha=0.4,
        label="Normal (0)",
    )
    plt.scatter(
        fail["tool_wear_min"],
        fail["torque_nm"],
        alpha=0.8,
        label="Failure (1)",
    )
    plt.xlabel("Tool wear [min]")
    plt.ylabel("Torque [Nm]")
    plt.title("Torque vs Tool Wear by Failure Status")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    # print(f"Loading dataset from {DATA_PATH}...")
    df = load_ai4i("ai4i2020.csv")

    make_output_dir(OUTPUT_DIR)

    # Class balance plot
    y = df["machine_failure"]
    plot_class_balance(y, os.path.join(OUTPUT_DIR, "class_balance.png"))

    # Prepare features/labels
    feature_cols = [
        "air_temp_k",
        "process_temp_k",
        "temp_diff_k",
        "rot_speed_rpm",
        "torque_nm",
        "tool_wear_min",
        "type",
    ]
    X = df[feature_cols].copy()
    y = df["machine_failure"]

    X = pd.get_dummies(X, columns=["type"], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Train models
    results = train_models(X_train, y_train, X_test, y_test)

    # ROC & PR curves
    plot_roc_curves(results, y_test, os.path.join(OUTPUT_DIR, "roc_curves.png"))
    plot_pr_curves(results, y_test, os.path.join(OUTPUT_DIR, "pr_curves.png"))

    # Metrics comparison
    plot_metrics_comparison(
        results, os.path.join(OUTPUT_DIR, "metrics_comparison.png")
    )

    # Feature importance for XGBoost (last model in list)
    xgb_res = [r for r in results if r["name"] == "XGBoost"][0]
    xgb_model = xgb_res["model"]
    plot_feature_importance_xgb(
        xgb_model, list(X.columns), os.path.join(OUTPUT_DIR, "feature_importance_xgb.png")  # noqa: E501
    )

    # Confusion matrix for best model (by F1 for failure)
    best = max(results, key=lambda r: r["f1_fail"])
    print("\nBest model by F1 (failure):", best["name"])
    plot_confusion_matrix(
        y_test, best["y_pred"], os.path.join(OUTPUT_DIR, "confusion_matrix_best.png")  # noqa: E501
    )

    # Scatter of torque vs tool wear
    plot_torque_toolwear_scatter(
        df, os.path.join(OUTPUT_DIR, "torque_toolwear_scatter.png")
    )

    print("\nDone. Figures written to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()

