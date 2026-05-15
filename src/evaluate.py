"""
evaluate.py
-----------
Loads the saved results.json and produces four publication-quality plots:

    1. ROC curve
    2. Confusion matrix (normalised)
    3. Feature importance bar chart
    4. 5-fold cross-validation scores

Run:
    python src/evaluate.py

Plots are saved to models/plots/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_PATH = Path(__file__).parent.parent / "models" / "results.json"
PLOTS_DIR    = Path(__file__).parent.parent / "models" / "plots"

PALETTE = {
    "bg":      "#0f0e17",
    "card":    "#1a1928",
    "accent":  "#e8b86d",
    "purple":  "#9b8ec4",
    "green":   "#5fcf80",
    "red":     "#e05c5c",
    "muted":   "#7a78a0",
    "text":    "#e8e6f0",
}

FEATURE_LABELS = {
    "Pclass":       "Passenger class",
    "Sex_enc":      "Sex",
    "Age":          "Age",
    "SibSp":        "Siblings / spouses",
    "Parch":        "Parents / children",
    "Fare":         "Fare",
    "FamilySize":   "Family size",
    "IsAlone":      "Travelling alone",
    "FarePerPerson":"Fare per person",
    "Embarked_enc": "Port of embarkation",
    "AgeGroup_enc": "Age group",
}


def style_axes(ax, title=""):
    ax.set_facecolor(PALETTE["card"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE["muted"])
        spine.set_alpha(0.3)
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=12, fontweight="bold", pad=10)


def plot_roc(ax, fpr, tpr, auc):
    ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE["accent"])
    ax.plot(fpr, tpr, color=PALETTE["accent"], lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["muted"], lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False, labelcolor=PALETTE["text"], fontsize=9)
    style_axes(ax, "ROC Curve")


def plot_confusion(ax, cm):
    total = np.array(cm).sum(axis=1, keepdims=True)
    cm_norm = np.array(cm) / total  # row-normalised

    im = ax.imshow(cm_norm, cmap="Purples", vmin=0, vmax=1)
    labels = ["Died", "Survived"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Pred:\n{l}" for l in labels], color=PALETTE["text"])
    ax.set_yticklabels([f"Actual:\n{l}" for l in labels], color=PALETTE["text"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]}\n({cm_norm[i][j]:.0%})",
                    ha="center", va="center", fontsize=11,
                    color=PALETTE["text"] if cm_norm[i][j] < 0.6 else PALETTE["bg"],
                    fontweight="bold")

    style_axes(ax, "Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")


def plot_importances(ax, importances: dict):
    sorted_items = sorted(importances.items(), key=lambda x: x[1])
    names  = [FEATURE_LABELS.get(k, k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = [PALETTE["accent"] if v == max(values) else PALETTE["purple"] for v in values]

    bars = ax.barh(names, values, color=colors, height=0.65)
    ax.set_xlabel("Importance (mean decrease in Gini impurity)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    for bar, val in zip(bars, values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=8, color=PALETTE["muted"])
    style_axes(ax, "Feature Importances")


def plot_cv(ax, cv_scores: list):
    folds  = [f"Fold {i+1}" for i in range(len(cv_scores))]
    colors = [PALETTE["green"] if s >= np.mean(cv_scores) else PALETTE["red"] for s in cv_scores]
    bars   = ax.bar(folds, cv_scores, color=colors, width=0.5)
    ax.axhline(np.mean(cv_scores), color=PALETTE["accent"], lw=1.5, linestyle="--",
               label=f"Mean = {np.mean(cv_scores):.1%}")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.6, 0.9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(frameon=False, labelcolor=PALETTE["text"], fontsize=9)
    for bar, val in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9, color=PALETTE["text"])
    style_axes(ax, "5-Fold Cross-Validation")


def main():
    with open(RESULTS_PATH) as f:
        r = json.load(f)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Titanic Survival Predictor — Model Evaluation",
                 color=PALETTE["text"], fontsize=15, fontweight="bold", y=0.98)

    plot_roc(axes[0, 0], r["fpr"], r["tpr"], r["auc"])
    plot_confusion(axes[0, 1], r["cm"])
    plot_importances(axes[1, 0], r["importances"])
    plot_cv(axes[1, 1], r["cv_scores"])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = PLOTS_DIR / "evaluation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"Saved evaluation plot → {out}")
    plt.show()


if __name__ == "__main__":
    main()