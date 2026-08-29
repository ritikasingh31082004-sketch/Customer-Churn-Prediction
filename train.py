"""
train.py
--------
Trains and compares several classifiers for customer churn prediction,
selects the best model by ROC-AUC on a held-out test set, and saves:
  - models/churn_model.joblib   (full pipeline: preprocessing + model)
  - reports/model_comparison.csv
  - reports/figures/roc_curves.png
  - reports/figures/confusion_matrix.png
  - reports/figures/feature_importance.png

Run from project root:
    python src/train.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from preprocess import load_data, split_features_target, build_preprocessor

RANDOM_SEED = 42
DATA_PATH = "data/customer_churn.csv"
MODEL_PATH = "models/churn_model.joblib"
FIG_DIR = "reports/figures"


def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }


def main():
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor()
    results = []
    fitted_pipelines = {}

    plt.figure(figsize=(6, 5))

    for name, model in get_candidate_models().items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)

        test_auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)

        results.append(
            {
                "model": name,
                "cv_auc_mean": cv_scores.mean(),
                "cv_auc_std": cv_scores.std(),
                "test_auc": test_auc,
                "precision_churn": report["1"]["precision"],
                "recall_churn": report["1"]["recall"],
                "f1_churn": report["1"]["f1-score"],
                "accuracy": report["accuracy"],
            }
        )
        fitted_pipelines[name] = pipe

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={test_auc:.3f})")

        print(f"{name}: CV AUC={cv_scores.mean():.3f} (+/-{cv_scores.std():.3f}), "
              f"Test AUC={test_auc:.3f}")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/roc_curves.png", dpi=120)
    plt.close()

    results_df = pd.DataFrame(results).sort_values("test_auc", ascending=False)
    results_df.to_csv("reports/model_comparison.csv", index=False)
    print("\nModel comparison:\n", results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_pipe = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name}")

    y_pred_best = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=120)
    plt.close()

    # Feature importance (if the best model supports it)
    model_step = best_pipe.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        feature_names = best_pipe.named_steps["preprocess"].get_feature_names_out()
        importances = model_step.feature_importances_
        imp_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False).head(15)

        plt.figure(figsize=(7, 6))
        plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#4C72B0")
        plt.title(f"Top 15 Feature Importances - {best_name}")
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=120)
        plt.close()

    joblib.dump(best_pipe, MODEL_PATH)
    print(f"\nSaved best model pipeline to {MODEL_PATH}")

    with open("reports/best_model_metrics.json", "w") as f:
        json.dump(results_df.iloc[0].to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
