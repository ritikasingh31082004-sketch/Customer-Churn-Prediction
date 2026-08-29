"""
predict.py
----------
Loads the trained pipeline and scores new customer records for churn risk.

Usage:
    python src/predict.py --input path/to/new_customers.csv --output predictions.csv

The input CSV must contain the same feature columns used in training
(see preprocess.NUMERIC_FEATURES / CATEGORICAL_FEATURES). customer_id,
if present, is carried through to the output but not used as a feature.
"""

import argparse
import joblib
import pandas as pd

from preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ID_COLUMN

MODEL_PATH = "models/churn_model.joblib"


def predict(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    model = joblib.load(MODEL_PATH)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    X = df[feature_cols]
    churn_proba = model.predict_proba(X)[:, 1]
    churn_pred = model.predict(X)

    out = df.copy()
    out["churn_probability"] = churn_proba.round(4)
    out["churn_prediction"] = pd.Series(churn_pred).map({1: "Yes", 0: "No"})

    cols_order = ([ID_COLUMN] if ID_COLUMN in out.columns else []) + \
        feature_cols + ["churn_probability", "churn_prediction"]
    out = out[cols_order]

    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} predictions to {output_path}")
    print(out[["churn_probability", "churn_prediction"]].describe(include="all"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict customer churn.")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="predictions.csv", help="Path to output CSV")
    args = parser.parse_args()
    predict(args.input, args.output)
