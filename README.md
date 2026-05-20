# 🚢 Titanic Survival Predictor

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Titanic-blue)
![Paradigm](https://img.shields.io/badge/ML-Supervised%20Classification-purple)
![Model](https://img.shields.io/badge/Model-Random%20Forest-orange)

A supervised machine-learning project that predicts whether a Titanic passenger
survived, using a **Random Forest classifier** built from scratch in Python.

---

## What it does

Given a passenger's details — class, sex, age, fare, family size, and port of
embarkation — the model outputs a survival probability and a verdict. It also
compares three classifiers, visualises model performance, and supports both
interactive and batch prediction.

**Results:**

| Metric | Value |
|---|---|
| Test accuracy | ~82% |
| ROC-AUC | ~0.87 |
| 5-fold CV mean | ~83% ± 2% |

The single most predictive feature is **sex** (~38% importance), reflecting the
historical "women and children first" policy. Fare and class together add another
~24%, encoding deck proximity and socioeconomic status.

---

## The dataset

This project uses the **real Kaggle Titanic dataset** — 891 actual passengers
from the 1912 disaster, downloaded automatically from a public GitHub mirror.
No synthetic data is generated.

Note: the 891 rows represent only the passengers whose records survived. The
Titanic carried ~2,208 people in total; the rest have incomplete or lost
documentation. The real historical survival rate was ~31.6%, slightly lower than
the 38.4% in the dataset due to survivorship bias toward wealthier passengers
with better records.

---

## Run order

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the real dataset

```bash
python src/load_data.py
```

Downloads the real Titanic passenger data to `data/titanic.csv` from a public
mirror of the Kaggle competition dataset. If the download fails (e.g. firewall),
grab `train.csv` manually from [kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data)
and save it as `data/titanic.csv`.

### 3. Train the model

```bash
python src/train.py
```

Loads the CSV, engineers features, cross-validates three models (Random Forest,
Gradient Boosting, Logistic Regression), evaluates the best on a held-out test
set, and saves `models/random_forest.pkl` and `models/results.json`.

### 4a. Predict interactively

```bash
python src/predict.py
```

Prompts you for passenger details and returns a survival probability. Age can
be left blank if unknown — the model handles missing ages natively.

```
Passenger class: 1
Sex: female
Age (leave blank if unknown): 32
Fare: £80

Survival probability: 94.1%
Verdict: Survived
```

### 4b. Predict in batch

```bash
python src/predict.py --csv path/to/passengers.csv
```

Scores a whole CSV of passengers and saves `_predictions.csv`.

### 5. Evaluate and plot

```bash
python src/evaluate.py
```

Reads `models/results.json` and saves a 4-panel chart to
`models/plots/evaluation.png` showing the ROC curve, confusion matrix,
feature importances, and cross-validation scores.

```bash
xdg-open models/plots/evaluation.png   # Linux
open models/plots/evaluation.png       # macOS
```

---

## License

MIT