"""
train.py
--------
Trains three classifiers on the real Titanic dataset, compares via 5-fold CV,
evaluates the best on a held-out test set, and saves model + results.

Run AFTER load_data.py:
    python src/train.py

Outputs:
    models/random_forest.pkl   — model bundle
    models/results.json        — metrics, confusion matrix, ROC, importances
"""

import json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from features import engineer_features, FEATURE_COLS

DATA_PATH    = Path(__file__).parent.parent / "data" / "titanic.csv"
MODELS_DIR   = Path(__file__).parent.parent / "models"
MODEL_PATH   = MODELS_DIR / "random_forest.pkl"
RESULTS_PATH = MODELS_DIR / "results.json"
TEST_SIZE    = 0.20
SEED         = 42


def load():
    if not DATA_PATH.exists():
        print("Data not found. Run `python src/load_data.py` first.")
        raise SystemExit(1)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} real passengers")
    print(f"  Age missing: {df['Age'].isna().sum()} (kept as NaN — historically lost)")
    X, y, title_enc = engineer_features(df)
    return X.values, y.values, title_enc


def main():
    X, y, title_enc = load()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                               random_state=SEED, stratify=y)
    print(f"Train: {len(X_tr)}  |  Test: {len(X_te)}\n")

    models = {
        # RandomForest handles NaN natively
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                         min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        # HistGradientBoosting handles NaN natively
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=300,
                         max_depth=4, learning_rate=0.05, random_state=SEED),
        # Logistic Regression does not — impute only for this one
        "Logistic Regression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
    }

    print(f"{'Model':<25} {'CV Acc':>8}  {'Std':>7}")
    print("-" * 45)
    cv_scores_rf = None
    for name, m in models.items():
        cv = cross_val_score(m, X, y, cv=5, scoring="accuracy")
        print(f"{name:<25} {cv.mean():.3f}     ±{cv.std():.3f}")
        if name == "Random Forest":
            cv_scores_rf = cv

    rf = models["Random Forest"]
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    cm  = confusion_matrix(y_te, y_pred)
    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)

    print(f"\nTest Accuracy: {acc:.4f}  |  ROC-AUC: {auc:.4f}")
    print(f"Confusion matrix:\n{cm}")
    print(classification_report(y_te, y_pred))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": rf, "features": FEATURE_COLS,
                     "title_encoder": title_enc}, f)

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "acc": round(acc,4), "auc": round(auc,4),
            "cm": cm.tolist(), "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "cv_scores": [round(s,4) for s in cv_scores_rf],
            "features": FEATURE_COLS,
            "importances": dict(zip(FEATURE_COLS, rf.feature_importances_.tolist())),
            "train_size": len(X_tr), "test_size": len(X_te),
        }, f, indent=2)

    print(f"\nSaved → {MODEL_PATH}")
    print(f"Saved → {RESULTS_PATH}")

if __name__ == "__main__":
    main()