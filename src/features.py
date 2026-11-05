"""Leakage-safe feature engineering and preprocessing."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGETS = ["final_score", "at_risk"]
DROP_COLUMNS = ["student_id", "final_score", "at_risk"]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["study_attendance_interaction"] = (
            X["study_hours_per_day"] * X["attendance_pct"] / 100
        )
        X["assessment_average"] = (
            X["assignment_completion_pct"] + X["internal_marks"]
        ) / 2
        X["deadline_rate"] = X["missed_deadlines"] / (X["semester"] + 1)
        X["engagement_per_study_hour"] = X["lms_logins"] / (X["study_hours_per_day"] + 0.5)
        X["commute_burden"] = X["commute_minutes"] / 60
        X["sleep_deviation"] = np.abs(X["sleep_hours"] - 8)
        return X

def make_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical)
    ])
