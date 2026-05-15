"""
features.py
-----------
Feature engineering for the real Titanic dataset.

Real data has columns our synthetic set didn't:
  - Name    → extract Title (Mr, Mrs, Miss, Master, Rare)
  - Cabin   → HasCabin flag (cabin assignment correlates with class/deck)
  - Ticket  → not used (too sparse to generalise)

Age is intentionally left as NaN where unknown. The 177 missing ages are
genuinely lost to history — imputing them would fabricate data. Both
RandomForest and HistGradientBoosting handle NaN natively.

Usage:
    from src.features import engineer_features, FEATURE_COLS

    df_raw     = pd.read_csv("data/titanic.csv")
    X, y, enc  = engineer_features(df_raw)
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [
    "Pclass", "Sex_enc", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone", "FarePerPerson",
    "Embarked_enc", "AgeGroup_enc",
    "Title_enc", "HasCabin",
]


def extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", str(name))
    if not match:
        return "Rare"
    title = match.group(1).strip()
    rare = {"Capt","Col","Don","Dr","Jonkheer","Lady","Major","Rev","Sir","Countess","Dona"}
    if title in rare:        return "Rare"
    if title in {"Mlle","Ms"}: return "Miss"
    if title == "Mme":       return "Mrs"
    return title


def engineer_features(
    df: pd.DataFrame,
    title_encoder: LabelEncoder | None = None,
) -> tuple:
    """
    Transform a raw Titanic dataframe into model-ready features.

    Age is left as NaN where unknown — missing ages are historically lost,
    not a data quality error we should paper over. The models handle NaN
    natively so no information is fabricated.

    Embarked: only 2 values missing, both historically confirmed as
    Southampton, so filled with 'S'.
    Fare: only 1 value missing, filled with median as a safe default.

    Parameters
    ----------
    df            : Raw dataframe.
    title_encoder : Fitted LabelEncoder for Title (reuse at inference).

    Returns
    -------
    X             : DataFrame with exactly FEATURE_COLS columns.
    y             : Series of Survived labels, or None if not in df.
    title_encoder : The fitted LabelEncoder — save this in the model bundle.
    """
    df = df.copy()

    # ── Minimal imputation — only where historically justified ─────────────
    df["Embarked"] = df["Embarked"].fillna("S")          # 2 missing, confirmed Southampton
    df["Fare"]     = df["Fare"].fillna(df["Fare"].median())  # 1 missing value

    # Age intentionally left as NaN

    # ── Core derived features ───────────────────────────────────────────────
    df["FamilySize"]    = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]       = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # NaN ages produce a NaN AgeGroup — encoded as -1, which the model treats
    # as its own implicit "unknown age" category
    df["AgeGroup"] = pd.cut(
        df["Age"], bins=[0,12,18,35,60,100],
        labels=["Child","Teen","Adult","MiddleAge","Senior"],
    )

    # ── Features unique to the real dataset ────────────────────────────────
    if "Name" in df.columns:
        df["Title"] = df["Name"].apply(extract_title)
    else:
        df["Title"] = "Mr"

    if "Cabin" in df.columns:
        df["HasCabin"] = df["Cabin"].notna().astype(int)
    else:
        df["HasCabin"] = 0

    # ── Encoding ────────────────────────────────────────────────────────────
    df["Sex_enc"]      = (df["Sex"] == "male").astype(int)
    df["Embarked_enc"] = LabelEncoder().fit_transform(df["Embarked"])

    # AgeGroup encoding: NaN → -1 (unknown age, treated as its own signal)
    age_cat = df["AgeGroup"].cat.codes  # NaN becomes -1 automatically
    df["AgeGroup_enc"] = age_cat

    if title_encoder is None:
        title_encoder = LabelEncoder()
        df["Title_enc"] = title_encoder.fit_transform(df["Title"])
    else:
        known = set(title_encoder.classes_)
        df["Title"] = df["Title"].apply(lambda t: t if t in known else "Rare")
        df["Title_enc"] = title_encoder.transform(df["Title"])

    X = df[FEATURE_COLS]
    y = df["Survived"] if "Survived" in df.columns else None
    return X, y, title_encoder