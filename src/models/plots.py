import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pathlib import Path

RESULTS_PATH = Path("model_results.json")
FIGURES_DIR  = Path("reports/modeling/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Random Forest":      "#378ADD",
    "XGBoost":            "#1D9E75",
    "Logistic Regression":"#888780",
}
MODEL_ORDER = ["Random Forest", "XGBoost", "Logistic Regression"]


def _load(path: Path = RESULTS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


# ── 1. Grouped metrics bar chart ───────────────────────────────────────────

def plot_metrics_comparison(results: dict, save: bool = True) -> plt.Figure:
    """
    Grouped bar chart: Accuracy, F1, Precision, Recall for each model.
    """
    metric_keys   = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    models  = MODEL_ORDER
    n_m     = len(metric_keys)
    n_mod   = len(models)
    x       = np.arange(n_m)
    width   = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [results["models"][model][k] for k in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=model,
                      color=COLORS[model], alpha=0.88, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0.55, 1.03)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model performance — core metrics", fontsize=13, pad=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "01_metrics_comparison.png", dpi=150)
    return fig


# ── 2. ROC-AUC bar chart ───────────────────────────────────────────────────

def plot_roc_auc(results: dict, save: bool = True) -> plt.Figure:
    """
    Horizontal bar chart of ROC-AUC scores with annotation.
    """
    models = MODEL_ORDER
    vals   = [results["models"][m]["roc_auc"] for m in models]
    colors = [COLORS[m] for m in models]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(models, vals, color=colors, alpha=0.88, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=10)

    ax.set_xlim(0.6, 1.05)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("ROC-AUC comparison", fontsize=13, pad=10)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "02_roc_auc.png", dpi=150)
    return fig


# ── 3. Holiday accuracy vs regular accuracy ────────────────────────────────

def plot_holiday_vs_regular(results: dict, save: bool = True) -> plt.Figure:
    """
    Side-by-side bars comparing holiday accuracy vs regular accuracy per model.
    """
    models = MODEL_ORDER
    hol    = [results["models"][m]["holiday_accuracy"] for m in models]
    reg    = [results["models"][m]["regular_accuracy"]  for m in models]

    x     = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - width / 2, reg, width, label="Regular weeks",
                color=[COLORS[m] for m in models], alpha=0.88, zorder=3)
    b2 = ax.bar(x + width / 2, hol, width, label="Holiday weeks",
                color=[COLORS[m] for m in models], alpha=0.45, zorder=3,
                hatch="///", edgecolor="white", linewidth=0.5)

    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0.6, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Holiday vs regular week accuracy", fontsize=13, pad=10)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "03_holiday_vs_regular.png", dpi=150)
    return fig


# ── 4. Weighted classification error ──────────────────────────────────────

def plot_wce(results: dict, save: bool = True) -> plt.Figure:
    """
    Bar chart of weighted classification error (lower is better).
    """
    models = MODEL_ORDER
    vals   = [results["models"][m]["weighted_classification_error"] for m in models]
    colors = [COLORS[m] for m in models]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(models, vals, color=colors, alpha=0.88, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 0.40)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Weighted classification error (lower = better)", fontsize=13, pad=10)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "04_wce.png", dpi=150)
    return fig


# ── 5. Feature importance — grouped horizontal bars ───────────────────────

def plot_feature_importance_grouped(results: dict, save: bool = True) -> plt.Figure:
    """
    Grouped horizontal bar chart of feature importance for all three models.
    """
    fi      = results["feature_importance"]
    features = list(fi["Random Forest"].keys())
    models   = MODEL_ORDER

    n_f   = len(features)
    n_mod = len(models)
    y     = np.arange(n_f)
    height = 0.25

    fig, ax = plt.subplots(figsize=(11, 7))
    for i, model in enumerate(models):
        vals = [fi[model].get(f, 0) for f in features]
        ax.barh(y + i * height, vals, height, label=model,
                color=COLORS[model], alpha=0.88, zorder=3)

    ax.set_yticks(y + height)
    ax.set_yticklabels(features, fontsize=10)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Feature importance — all models", fontsize=13, pad=10)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "05_feature_importance_grouped.png", dpi=150)
    return fig


# ── 6. Per-model feature importance (sorted) ──────────────────────────────

def plot_feature_importance_per_model(results: dict, save: bool = True) -> plt.Figure:
    """
    One horizontal bar chart per model, sorted by importance descending.
    """
    fi     = results["feature_importance"]
    models = MODEL_ORDER

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, model in zip(axes, models):
        items  = sorted(fi[model].items(), key=lambda x: x[1], reverse=True)
        labels = [k for k, _ in items]
        vals   = [v for _, v in items]

        bars = ax.barh(labels[::-1], vals[::-1], color=COLORS[model], alpha=0.88, zorder=3)
        for bar, v in zip(bars, vals[::-1]):
            ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8)

        ax.set_title(model, fontsize=11, color=COLORS[model], pad=8)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis="x", alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Feature importance by model (sorted)", fontsize=13, y=1.01)
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "06_feature_importance_per_model.png",
                    dpi=150, bbox_inches="tight")
    return fig


# ── 7. Radar / spider chart ────────────────────────────────────────────────

def plot_radar(results: dict, save: bool = True) -> plt.Figure:
    """
    Radar chart comparing models across 5 normalized metrics.
    """
    metric_keys   = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    models = MODEL_ORDER

    angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for model in models:
        vals  = [results["models"][model][k] for k in metric_keys]
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[model], linewidth=2, label=model)
        ax.fill(angles, vals, color=COLORS[model], alpha=0.10)

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title("Model comparison — radar view", fontsize=13, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False, fontsize=10)
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / "07_radar.png", dpi=150, bbox_inches="tight")
    return fig


# ── Master call ────────────────────────────────────────────────────────────

def plot_all(path: Path = RESULTS_PATH) -> None:
    """
    Load results JSON and generate all seven figures to FIGURES_DIR.
    """
    results = _load(path)
    plot_metrics_comparison(results)
    plot_roc_auc(results)
    plot_holiday_vs_regular(results)
    plot_wce(results)
    plot_feature_importance_grouped(results)
    plot_feature_importance_per_model(results)
    plot_radar(results)
    print(f"[✓] 7 figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    plot_all()