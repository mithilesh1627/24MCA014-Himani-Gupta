"""Evaluate generalization to institutions unseen during training."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import DATA_RAW, REPORTS, RANDOM_STATE
from src.features import FeatureEngineer, make_preprocessor, DROP_COLUMNS
from sklearn.pipeline import Pipeline

def main():
    df = pd.read_csv(DATA_RAW / "synthetic_students.csv")
    X = df.drop(columns=DROP_COLUMNS)
    y = df["final_score"]
    # Institutions 0-7 are source domains; institutions 8-9 are held-out target domains.
    train_mask = df["institution_id"] < 8
    test_mask = df["institution_id"] >= 8
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Domain ID is excluded from features so the model must learn transferable patterns.
    X_train = X_train.drop(columns=["institution_id"])
    X_test = X_test.drop(columns=["institution_id"])
    prep = make_preprocessor(FeatureEngineer().fit_transform(X_train))
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=120, max_depth=18,
                                              min_samples_leaf=4, n_jobs=-1,
                                              random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=180,
                                                              max_leaf_nodes=25,
                                                              random_state=RANDOM_STATE)
    }
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("features", FeatureEngineer()),
                         ("preprocess", prep), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({"model": name, "evaluation": "unseen_institutions_8_9",
                     "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                     "R2": r2_score(y_test, pred),
                     "train_rows": len(X_train), "test_rows": len(X_test)})
    out = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True, parents=True)
    out.to_csv(REPORTS / "domain_generalization.csv", index=False)
    print(out.to_string(index=False))
    print("\\nInterpretation: this is a held-out-domain stress test, not proof of universal fairness.")
if __name__ == "__main__":
    main()
