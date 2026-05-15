"""
predict.py
----------
Load the saved model and predict survival for one or more passengers.

Usage — interactive:
    python src/predict.py

Usage — batch CSV:
    python src/predict.py --csv path/to/passengers.csv

Batch CSV must have: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.
Adding a Name column enables title-based features (Mr, Mrs, Master, etc.).
Age can be left blank — the model handles unknown ages natively.
"""

import argparse, pickle
import numpy as np
from pathlib import Path
import pandas as pd
from features import engineer_features

MODEL_PATH     = Path(__file__).parent.parent / "models" / "random_forest.pkl"
VALID_SEX      = {"male", "female"}
VALID_EMBARKED = {"S", "C", "Q"}
VALID_PCLASS   = {1, 2, 3}


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_df(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    X, _, _ = engineer_features(df, title_encoder=bundle.get("title_encoder"))
    probs    = bundle["model"].predict_proba(X.values)[:, 1]
    survived = (probs >= 0.5).astype(int)
    out = df[["PassengerId"]].copy() if "PassengerId" in df.columns else pd.DataFrame()
    out["SurvivalProbability"] = probs.round(3)
    out["Prediction"]          = survived
    out["Verdict"]             = ["Survived" if s else "Did not survive" for s in survived]
    return out


def ask_int(prompt, default, valid_set):
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        val = raw if raw else str(default)
        try:
            result = int(val)
            if result not in valid_set:
                print(f"    Please enter one of: {sorted(valid_set)}")
                continue
            return result
        except ValueError:
            print(f"    Please enter a whole number.")


def ask_str(prompt, default, valid_set):
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        val = (raw if raw else default).strip().lower()
        # build a case-insensitive match
        match = next((v for v in valid_set if v.lower() == val), None)
        if match is None:
            print(f"    Please enter one of: {sorted(valid_set)}")
            continue
        return match


def ask_float(prompt, default):
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        val = raw if raw else str(default)
        try:
            return float(val)
        except ValueError:
            print("    Please enter a number.")


def ask_age():
    while True:
        raw = input("  Age (leave blank if unknown) [unknown]: ").strip()
        if not raw:
            return np.nan
        try:
            age = float(raw)
            if age <= 0 or age > 120:
                print("    Please enter a realistic age between 1 and 120.")
                continue
            return age
        except ValueError:
            print("    Please enter a number or leave blank.")


def interactive_predict(bundle: dict):
    print("\n── Titanic Survival Predictor ──────────────────────────────")
    print("Enter passenger details (press Enter for defaults):\n")

    pclass   = ask_int("Passenger class (1 / 2 / 3)", 3, VALID_PCLASS)
    name     = input("  Name (e.g. Mr. John Smith) [Mr. John Smith]: ").strip() or "Mr. John Smith"
    sex      = ask_str("Sex (male / female)", "male", VALID_SEX)
    age      = ask_age()
    sibsp    = ask_int("Siblings/spouses aboard", 0, set(range(9)))
    parch    = ask_int("Parents/children aboard", 0, set(range(7)))
    fare     = ask_float("Fare paid (£)", 15)
    embarked = ask_str("Port of embarkation (S / C / Q)", "S", VALID_EMBARKED)

    passenger = {
        "PassengerId": 9999,
        "Pclass": pclass, "Name": name, "Sex": sex, "Age": age,
        "SibSp": sibsp, "Parch": parch, "Fare": fare,
        "Embarked": embarked, "Cabin": None,
    }

    df = pd.DataFrame([passenger])
    result = predict_df(df, bundle).iloc[0]
    age_str = f"{age:.0f}" if not np.isnan(age) else "unknown"

    print(f"\n── Result ──────────────────────────────────────────────────")
    print(f"Age used             : {age_str}")
    print(f"Survival probability : {result['SurvivalProbability']:.1%}")
    print(f"Verdict              : {result['Verdict']}")
    print("─────────────────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to a CSV of passengers to score")
    args = parser.parse_args()

    bundle = load_model()
    print(f"Model loaded from {MODEL_PATH}")

    if args.csv:
        df = pd.read_csv(args.csv)
        results = predict_df(df, bundle)
        print(results.to_string(index=False))
        out = Path(args.csv).stem + "_predictions.csv"
        results.to_csv(out, index=False)
        print(f"\nSaved → {out}")
    else:
        interactive_predict(bundle)

if __name__ == "__main__":
    main()