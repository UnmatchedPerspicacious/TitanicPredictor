"""
load_data.py
------------
Downloads the real Titanic dataset and saves it to data/titanic.csv.

Source: GitHub (datasciencedojo) — mirrors the official Kaggle competition data.

Run:
    python src/load_data.py
"""

import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

OUTPUT = Path(__file__).parent.parent / "data" / "titanic.csv"
URL    = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

EXPECTED_COLS = {"Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"}

COL_MAP = {
    "survived":    "Survived",
    "pclass":      "Pclass",
    "sex":         "Sex",
    "age":         "Age",
    "sibsp":       "SibSp",
    "parch":       "Parch",
    "fare":        "Fare",
    "embarked":    "Embarked",
    "name":        "Name",
    "ticket":      "Ticket",
    "cabin":       "Cabin",
    "passengerid": "PassengerId",
}


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [COL_MAP.get(c.lower().strip(), c) for c in df.columns]
    if "Survived" in df.columns:
        df["Survived"] = pd.to_numeric(df["Survived"], errors="coerce").astype("Int64")
    return df


def validate(df: pd.DataFrame) -> bool:
    return EXPECTED_COLS.issubset(set(df.columns))


def download() -> pd.DataFrame:
    print(f"Downloading from {URL} …")
    try:
        resp = requests.get(URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df = normalise(df)
        if not validate(df):
            raise ValueError(f"Unexpected columns: {df.columns.tolist()}")
        print(f"  Downloaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("Download the file manually from:")
        print("  https://www.kaggle.com/competitions/titanic/data")
        print(f"and save it as: {OUTPUT}")
        sys.exit(1)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        print(f"Data already exists at {OUTPUT} — skipping download.")
        print("Delete the file and re-run to refresh.")
        df = pd.read_csv(OUTPUT)
    else:
        df = download()
        df.to_csv(OUTPUT, index=False)
        print(f"Saved to {OUTPUT}")

    print(f"\nDataset summary:")
    print(f"  Rows:          {len(df)}")
    print(f"  Columns:       {df.columns.tolist()}")
    print(f"  Survival rate: {df['Survived'].mean():.1%}")
    print(f"  Missing Age:   {df['Age'].isna().sum()} ({df['Age'].isna().mean():.0%})")
    print(f"  Missing Cabin: {df['Cabin'].isna().sum() if 'Cabin' in df.columns else 'N/A'}")


if __name__ == "__main__":
    main()