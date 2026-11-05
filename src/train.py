"""Train baseline, nonlinear, tuned and ensemble models."""
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, VotingRegressor, StackingRegressor,
    VotingClassifier, StackingClassifier, RandomForestRegressor)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from src.config import DATA_RAW, ARTIFACTS, REPORTS, RANDOM_STATE
from src.features import FeatureEngineer, make_preprocessor, DROP_COLUMNS

warnings.filterwarnings("ignore")

def regression_metrics(y, pred):
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred))}

def classification_metrics(y, pred, prob):
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, prob))}

def build_pipe(model, X):
    return Pipeline([("features", FeatureEngineer()),
                     ("preprocess", make_preprocessor(FeatureEngineer().fit_transform(X))),
                     ("model", model)])

def main():
    ARTIFACTS.mkdir(exist_ok=True, parents=True)
    REPORTS.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(DATA_RAW / "synthetic_students.csv")
    X = df.drop(columns=DROP_COLUMNS)
    y_reg, y_clf = df["final_score"], df["at_risk"]
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_reg, y_clf, test_size=.20, random_state=RANDOM_STATE, stratify=y_clf
    )

    # A compact tuning subset keeps the demo practical on a laptop.
    tune_n = min(18_000, len(X_train))
    tune_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_train), tune_n, replace=False)
    Xt, yt = X_train.iloc[tune_idx], yr_train.iloc[tune_idx]

    reg_models = {
        "LinearRegression_baseline": LinearRegression(),
        "Ridge": Ridge(alpha=2.0),
        "DecisionTree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=8, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_leaf=3,
                                             n_jobs=-1, random_state=RANDOM_STATE),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, max_depth=20, min_samples_leaf=3,
                                          n_jobs=-1, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, learning_rate=.08,
                                                              max_leaf_nodes=25, random_state=RANDOM_STATE)
    }
    clf_models = {
        "LogisticRegression_baseline": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=8, class_weight="balanced",
                                               random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=18, min_samples_leaf=3,
                                               class_weight="balanced_subsample", n_jobs=-1,
                                               random_state=RANDOM_STATE),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, max_depth=20, min_samples_leaf=3,
                                           class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=.08,
                                                               max_leaf_nodes=25, random_state=RANDOM_STATE)
    }

    reg_results, clf_results, fitted_reg, fitted_clf = [], [], {}, {}
    print("\\n[1/5] Training regression models...")
    for name, model in reg_models.items():
        pipe = build_pipe(model, X_train)
        pipe.fit(X_train, yr_train)
        pred = pipe.predict(X_test)
        reg_results.append({"model": name, **regression_metrics(yr_test, pred)})
        fitted_reg[name] = pipe
        print(name, reg_results[-1])

    print("\\n[2/5] Training classification models...")
    for name, model in clf_models.items():
        pipe = build_pipe(model, X_train)
        pipe.fit(X_train, yc_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= .5).astype(int)
        clf_results.append({"model": name, **classification_metrics(yc_test, pred, prob)})
        fitted_clf[name] = pipe
        print(name, clf_results[-1])

    print("\\n[3/5] Hyperparameter tuning (RandomizedSearchCV)...")
    tune_pipe = build_pipe(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), Xt)
    params = {
        "model__n_estimators": [80, 120, 160],
        "model__max_depth": [12, 18, 24, None],
        "model__min_samples_leaf": [2, 3, 5, 8],
        "model__max_features": [0.7, 1.0]
    }
    search = RandomizedSearchCV(tune_pipe, params, n_iter=5, cv=3,
                                scoring="neg_root_mean_squared_error",
                                random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
    search.fit(Xt, yt)
    tuned = build_pipe(search.best_estimator_.named_steps["model"], X_train)
    tuned.fit(X_train, yr_train)
    tuned_pred = tuned.predict(X_test)
    reg_results.append({"model": "TunedRandomForest", **regression_metrics(yr_test, tuned_pred)})
    fitted_reg["TunedRandomForest"] = tuned
    print("Best params:", search.best_params_)

    print("\\n[4/5] Fusion: voting and stacking ensembles...")
    # Diverse tree learners are fused; stacking learns how to combine their predictions.
    voting_reg = VotingRegressor([
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=18, n_jobs=-1, random_state=1)),
        ("et", ExtraTreesRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=2)),
        ("hgb", HistGradientBoostingRegressor(max_iter=150, random_state=3))
    ])
    vote_pipe = build_pipe(voting_reg, X_train)
    vote_pipe.fit(X_train, yr_train)
    reg_results.append({"model": "VotingFusion", **regression_metrics(yr_test, vote_pipe.predict(X_test))})
    fitted_reg["VotingFusion"] = vote_pipe

    stack_reg = StackingRegressor(estimators=[
        ("rf", RandomForestRegressor(n_estimators=80, max_depth=16, n_jobs=-1, random_state=1)),
        ("et", ExtraTreesRegressor(n_estimators=80, max_depth=18, n_jobs=-1, random_state=2)),
        ("hgb", HistGradientBoostingRegressor(max_iter=120, random_state=3))
    ], final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1)
    stack_pipe = build_pipe(stack_reg, X_train)
    stack_pipe.fit(X_train, yr_train)
    reg_results.append({"model": "StackingFusion", **regression_metrics(yr_test, stack_pipe.predict(X_test))})
    fitted_reg["StackingFusion"] = stack_pipe

    # Classification fusion with soft voting.
    vote_clf = VotingClassifier(estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=18, n_jobs=-1, random_state=1)),
        ("et", ExtraTreesClassifier(n_estimators=100, max_depth=18, n_jobs=-1, random_state=2)),
        ("hgb", HistGradientBoostingClassifier(max_iter=140, random_state=3))
    ], voting="soft", weights=[2, 2, 1])
    vote_clf_pipe = build_pipe(vote_clf, X_train)
    vote_clf_pipe.fit(X_train, yc_train)
    p = vote_clf_pipe.predict_proba(X_test)[:, 1]
    clf_results.append({"model": "SoftVotingFusion",
                        **classification_metrics(yc_test, (p >= .5).astype(int), p)})
    fitted_clf["SoftVotingFusion"] = vote_clf_pipe

    stack_clf = StackingClassifier(estimators=[
        ("rf", RandomForestClassifier(n_estimators=70, max_depth=16, n_jobs=-1, random_state=1)),
        ("et", ExtraTreesClassifier(n_estimators=70, max_depth=16, n_jobs=-1, random_state=2)),
        ("hgb", HistGradientBoostingClassifier(max_iter=120, random_state=3))
    ], final_estimator=LogisticRegression(max_iter=500), stack_method="predict_proba", cv=3, n_jobs=-1)
    stack_clf_pipe = build_pipe(stack_clf, X_train)
    stack_clf_pipe.fit(X_train, yc_train)
    p = stack_clf_pipe.predict_proba(X_test)[:, 1]
    clf_results.append({"model": "StackingFusion",
                        **classification_metrics(yc_test, (p >= .5).astype(int), p)})
    fitted_clf["StackingFusion"] = stack_clf_pipe

    reg_df = pd.DataFrame(reg_results).sort_values("RMSE")
    clf_df = pd.DataFrame(clf_results).sort_values("f1", ascending=False)
    reg_df.to_csv(REPORTS / "regression_model_comparison.csv", index=False)
    clf_df.to_csv(REPORTS / "classification_model_comparison.csv", index=False)

    best_reg_name = reg_df.iloc[0]["model"]
    best_clf_name = clf_df.iloc[0]["model"]
    joblib.dump(fitted_reg[best_reg_name], ARTIFACTS / "best_score_model.joblib")
    joblib.dump(fitted_clf[best_clf_name], ARTIFACTS / "best_risk_model.joblib")

    with open(REPORTS / "training_summary.json", "w") as f:
        json.dump({"best_regression_model": best_reg_name,
                   "best_classification_model": best_clf_name,
                   "tuning_best_params": search.best_params_,
                   "n_records": int(len(df)),
                   "test_size": .2}, f, indent=2)

    print("\\n[5/5] Done.")
    print("\\nRegression leaderboard:\\n", reg_df.to_string(index=False))
    print("\\nClassification leaderboard:\\n", clf_df.to_string(index=False))
    print(f"Saved best models: {best_reg_name}, {best_clf_name}")

if __name__ == "__main__":
    main()
