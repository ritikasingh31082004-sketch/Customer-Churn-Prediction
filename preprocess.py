"""
preprocess.py
-------------
Shared preprocessing pipeline for training and inference, built with
sklearn's ColumnTransformer so the exact same transformation is applied
at train time and prediction time (avoids train/serve skew).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = [
    "senior_citizen",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
]

CATEGORICAL_FEATURES = [
    "gender",
    "contract",
    "internet_service",
    "online_security",
    "tech_support",
    "paperless_billing",
    "payment_method",
]

TARGET = "churn"
ID_COLUMN = "customer_id"


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor
