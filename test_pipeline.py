"""
test_pipeline.py
-----------------
Lightweight sanity tests for the data and modeling pipeline.

Run from project root:
    python -m pytest tests/
"""

import sys
import os
import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import load_data, split_features_target, build_preprocessor

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "customer_churn.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.joblib")


def test_data_loads_and_has_expected_columns():
    df = load_data(DATA_PATH)
    expected = {"customer_id", "tenure_months", "contract", "monthly_charges", "churn"}
    assert expected.issubset(set(df.columns))
    assert len(df) > 0


def test_no_missing_values():
    df = load_data(DATA_PATH)
    assert df.isnull().sum().sum() == 0


def test_target_is_binary():
    df = load_data(DATA_PATH)
    assert set(df["churn"].unique()) == {"Yes", "No"}


def test_split_features_target_shapes():
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)
    assert len(X) == len(y) == len(df)
    assert y.isin([0, 1]).all()


def test_preprocessor_builds():
    preprocessor = build_preprocessor()
    assert preprocessor is not None


def test_saved_model_predicts():
    if not os.path.exists(MODEL_PATH):
        return  # model not trained yet in this environment; skip
    model = joblib.load(MODEL_PATH)
    df = load_data(DATA_PATH).sample(5, random_state=0)
    X, _ = split_features_target(df)
    preds = model.predict(X)
    probas = model.predict_proba(X)
    assert len(preds) == 5
    assert probas.shape == (5, 2)
