import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

from src.utils.logger import logger

matplotlib.use("Agg")

# Paths
EDA_DIR = Path("reports/eda")
FIGURES_DIR = EDA_DIR / "figures"
SUMMARY_PATH = EDA_DIR / "eda_summary.json"
DASHBOARD_DATA_PATH = EDA_DIR / "dashboard_data.json"

# Style Configuration
PALETTE = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "success": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
    "light": "#F3F4F6",
    "dark": "#1F2937",
}

CLASS_COLORS = {"Low": "#3B82F6", "High": "#EF4444"}
CLASS_PALETTE = [CLASS_COLORS["Low"], CLASS_COLORS["High"]]
TYPE_COLORS = {"A": "#2563EB", "B": "#7C3AED", "C": "#059669"}

FIGSIZE_WIDE = (16, 8)
FIGSIZE_SQUARE = (10, 10)
FIGSIZE_TALL = (14, 12)
FIGSIZE_SMALL = (8, 6)
DPI = 150
TITLE_SIZE = 14
LABEL_SIZE = 11

MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

CONTINUOUS_FEATURES = [
    "Weekly_Sales",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Size",
    "UMCSENT",
    "RSXFS",
    "PCE",
    "TotalMarkDown",
    "EconIndex",
    "ConsumerConfRatio",
    "FuelBurden",
    "PurchasingPower",
]

KEY_FEATURES = [
    "Size",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "TotalMarkDown",
    "EconIndex",
    "ActiveMarkDownCount",
    "HolidayProximity",
    "IsPeakSeason",
]


# Helpers
def _setup_style():
    """Configure matplotlib/seaborn for clean publication-ready plots."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CCCCCC",
            "grid.color": "#EEEEEE",
            "font.family": "sans-serif",
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
        }
    )


def _save_fig(fig: plt.Figure, name: str) -> Path:
    """Save figure and close to free memory."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("    Saved: {}", path.name)
    return path


def _safe_json(obj):
    """Convert numpy/pandas types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 4)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, pd.Period):
        return str(obj)
    return str(obj)


def _shape_str(df: pd.DataFrame) -> str:
    return f"{len(df):,} rows × {len(df.columns)} cols"


# GROUP 1: TARGET ANALYSIS
def plot_target_analysis(df: pd.DataFrame, report: dict) -> list[Path]:
    logger.info("Group 1: Target Analysis …")
    paths = []

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    counts = df["Sales_Label"].value_counts()
    bars = axes[0].bar(
        counts.index, counts.values, color=CLASS_PALETTE, edgecolor="white"
    )
    axes[0].set_title("Target Class Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1000,
            f"{val:,}\n({val / len(df) * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axes[1].pie(
        counts.values,
        labels=counts.index,
        colors=CLASS_PALETTE,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Class Proportion", fontweight="bold")

    for label, color in CLASS_COLORS.items():
        subset = df[df["Sales_Label"] == label]["Weekly_Sales"]
        axes[2].hist(subset, bins=80, alpha=0.6, label=label, color=color, density=True)
    axes[2].set_title("Weekly Sales Distribution by Class", fontweight="bold")
    axes[2].set_xlabel("Weekly Sales ($)")
    axes[2].set_ylabel("Density")
    axes[2].legend()
    axes[2].set_xlim(-5000, 80000)

    fig.suptitle("TARGET VARIABLE ANALYSIS", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    paths.append(_save_fig(fig, "01_target_distribution"))

    # ── 1b. Sales by class — box + violin ──
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    sns.boxplot(
        data=df, x="Sales_Label", y="Weekly_Sales", palette=CLASS_COLORS, ax=axes[0]
    )
    axes[0].set_title("Sales by Class (Box Plot)", fontweight="bold")
    axes[0].set_ylabel("Weekly Sales ($)")
    axes[0].set_ylim(-5000, 80000)

    sns.violinplot(
        data=df,
        x="Sales_Label",
        y="Weekly_Sales",
        palette=CLASS_COLORS,
        ax=axes[1],
        inner="quartile",
    )
    axes[1].set_title("Sales by Class (Violin Plot)", fontweight="bold")
    axes[1].set_ylabel("Weekly Sales ($)")
    axes[1].set_ylim(-5000, 80000)

    fig.tight_layout()
    paths.append(_save_fig(fig, "01_target_box_violin"))

    # ── 1c. Sales by class + store type ──
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.boxplot(
        data=df,
        x="Type",
        y="Weekly_Sales",
        hue="Sales_Label",
        palette=CLASS_COLORS,
        ax=ax,
        order=["A", "B", "C"],
    )
    ax.set_title("Weekly Sales by Store Type & Class", fontweight="bold")
    ax.set_ylabel("Weekly Sales ($)")
    ax.set_ylim(-5000, 80000)
    ax.legend(title="Sales Class")
    fig.tight_layout()
    paths.append(_save_fig(fig, "01_target_by_store_type"))

    # ── 1d. Sales statistics table as heatmap ──
    fig, ax = plt.subplots(figsize=(10, 4))
    stats_df = (
        df.groupby("Sales_Label")["Weekly_Sales"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(0)
    )
    stats_df.columns = ["Count", "Mean", "Median", "Std Dev", "Min", "Max"]
    ax.axis("off")
    table = ax.table(
        cellText=[[f"{v:,.0f}" for v in row] for row in stats_df.values],
        rowLabels=stats_df.index,
        colLabels=stats_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title("Sales Statistics by Class", fontweight="bold", pad=20)
    fig.tight_layout()
    paths.append(_save_fig(fig, "01_target_stats_table"))

    report["target_analysis"] = {
        "class_counts": counts.to_dict(),
        "class_pct": (counts / len(df) * 100).round(2).to_dict(),
        "sales_by_class": df.groupby("Sales_Label")["Weekly_Sales"]
        .agg(["mean", "median", "std"])
        .round(2)
        .to_dict(),
    }

    return paths


# GROUP 2: TEMPORAL PATTERNS
def plot_temporal_patterns(df: pd.DataFrame, report: dict) -> list[Path]:
    logger.info("Group 2: Temporal Patterns …")
    paths = []

    # ── 2a. Weekly sales over time (aggregated) ──
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    weekly = (
        df.groupby("Date")
        .agg(
            Total_Sales=("Weekly_Sales", "sum"),
            Avg_Sales=("Weekly_Sales", "mean"),
            High_Pct=("Sales_Class", "mean"),
        )
        .reset_index()
    )

    axes[0].plot(
        weekly["Date"], weekly["Avg_Sales"], color=PALETTE["primary"], linewidth=1.5
    )
    axes[0].fill_between(
        weekly["Date"], weekly["Avg_Sales"], alpha=0.15, color=PALETTE["primary"]
    )
    axes[0].set_title("Average Weekly Sales Over Time", fontweight="bold")
    axes[0].set_ylabel("Avg Weekly Sales ($)")

    # Highlight holidays
    holidays = df[df["IsHoliday"] is True]["Date"].unique()
    for h in holidays:
        axes[0].axvline(h, color=PALETTE["danger"], alpha=0.2, linewidth=0.8)

    axes[1].plot(
        weekly["Date"],
        weekly["High_Pct"] * 100,
        color=PALETTE["secondary"],
        linewidth=1.5,
    )
    axes[1].fill_between(
        weekly["Date"], weekly["High_Pct"] * 100, alpha=0.15, color=PALETTE["secondary"]
    )
    axes[1].set_title("% High-Sales Class Over Time", fontweight="bold")
    axes[1].set_ylabel("High Class %")
    axes[1].set_xlabel("Date")

    fig.suptitle("TEMPORAL SALES TRENDS", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    paths.append(_save_fig(fig, "02_temporal_trend"))

    # ── 2b. Monthly seasonality ──
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    monthly = df.groupby("Month")["Weekly_Sales"].agg(["mean", "median"]).reset_index()
    x = monthly["Month"]
    axes[0].bar(
        x - 0.2, monthly["mean"], width=0.4, label="Mean", color=PALETTE["primary"]
    )
    axes[0].bar(
        x + 0.2,
        monthly["median"],
        width=0.4,
        label="Median",
        color=PALETTE["secondary"],
    )
    axes[0].set_title("Monthly Average Sales", fontweight="bold")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Sales ($)")
    axes[0].set_xticks(range(1, 13))
    axes[0].legend()

    monthly_class = df.groupby("Month")["Sales_Class"].mean() * 100
    colors = [
        PALETTE["danger"] if v > 50 else PALETTE["primary"] for v in monthly_class
    ]
    axes[1].bar(monthly_class.index, monthly_class.values, color=colors)
    axes[1].axhline(50, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title("% High-Sales by Month", fontweight="bold")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("High Class %")
    axes[1].set_xticks(range(1, 13))

    fig.tight_layout()
    paths.append(_save_fig(fig, "02_monthly_seasonality"))

    # ── 2c. Holiday impact ──
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    if "HolidayName" in df.columns:
        holiday_sales = (
            df.groupby("HolidayName")["Weekly_Sales"]
            .agg(["mean", "count"])
            .reset_index()
        )
        holiday_sales = holiday_sales.sort_values("mean", ascending=True)
        colors = [
            PALETTE["danger"] if n != "None" else PALETTE["light"]
            for n in holiday_sales["HolidayName"]
        ]
        axes[0].barh(
            holiday_sales["HolidayName"],
            holiday_sales["mean"],
            color=colors,
            edgecolor="white",
        )
        axes[0].set_title("Average Sales by Holiday", fontweight="bold")
        axes[0].set_xlabel("Average Weekly Sales ($)")

        holiday_class = df.groupby("HolidayName")["Sales_Class"].mean() * 100
        holiday_class = holiday_class.sort_values()
        colors2 = [
            PALETTE["danger"] if v > 50 else PALETTE["primary"] for v in holiday_class
        ]
        axes[1].barh(
            holiday_class.index, holiday_class.values, color=colors2, edgecolor="white"
        )
        axes[1].axvline(50, color="gray", linestyle="--", alpha=0.5)
        axes[1].set_title("% High-Sales by Holiday", fontweight="bold")
        axes[1].set_xlabel("High Class %")

    fig.tight_layout()
    paths.append(_save_fig(fig, "02_holiday_impact"))

    # ── 2d. Day-of-year heatmap (Store × Week) ──
    fig, ax = plt.subplots(figsize=(18, 8))
    pivot = df.pivot_table(
        values="Weekly_Sales", index="Store", columns="Week", aggfunc="mean"
    )
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Avg Sales ($)"},
        xticklabels=4,
        yticklabels=5,
    )
    ax.set_title("Average Sales Heatmap: Store × Week of Year", fontweight="bold")
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("Store ID")
    fig.tight_layout()
    paths.append(_save_fig(fig, "02_store_week_heatmap"))

    # ── 2e. Pre/Post holiday effect ──
    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    period_data = []
    for col, label in [
        ("IsPreHoliday", "Pre-Holiday"),
        ("IsHoliday", "Holiday"),
        ("IsPostHoliday", "Post-Holiday"),
    ]:
        if col in df.columns:
            for val, state in [(1, label), (0, f"Non-{label}")]:
                sub = df[df[col] == val]["Weekly_Sales"]
                period_data.append({"Period": state, "Avg Sales": sub.mean()})

    if period_data:
        period_df = pd.DataFrame(period_data)
        period_df = period_df[
            period_df["Period"].isin(
                ["Pre-Holiday", "Holiday", "Post-Holiday", "Non-Holiday"]
            )
        ]
        if len(period_df) > 0:
            sns.barplot(
                data=period_df, x="Period", y="Avg Sales", palette="Set2", ax=ax
            )
            ax.set_title(
                "Sales: Pre-Holiday → Holiday → Post-Holiday", fontweight="bold"
            )
            ax.set_ylabel("Average Weekly Sales ($)")
    fig.tight_layout()
    paths.append(_save_fig(fig, "02_holiday_period_effect"))

    # ── 2f. Year-over-year comparison ──
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for year in sorted(df["Year"].unique()):
        yearly = df[df["Year"] == year].groupby("Week")["Weekly_Sales"].mean()
        ax.plot(yearly.index, yearly.values, label=str(year), linewidth=1.5)
    ax.set_title("Year-over-Year Weekly Sales Comparison", fontweight="bold")
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("Average Weekly Sales ($)")
    ax.legend(title="Year")
    fig.tight_layout()
    paths.append(_save_fig(fig, "02_yoy_comparison"))

    report["temporal_patterns"] = {
        "monthly_avg_sales": monthly["mean"].to_dict(),
        "holiday_avg_sales": (
            df.groupby("HolidayName")["Weekly_Sales"].mean().round(2).to_dict()
            if "HolidayName" in df.columns
            else {}
        ),
        "years_covered": sorted(df["Year"].unique().tolist()),
    }

    return paths


# GROUP 3: STORE ANALYSIS
def plot_store_analysis(df: pd.DataFrame, report: dict) -> list[Path]:
    logger.info("Group 3: Store Analysis …")
    paths = []

    # ── 3a. Store type comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(
        data=df,
        x="Type",
        y="Weekly_Sales",
        palette=TYPE_COLORS,
        ax=axes[0],
        order=["A", "B", "C"],
    )
    axes[0].set_title("Sales by Store Type", fontweight="bold")
    axes[0].set_ylim(-5000, 60000)

    type_counts = df.groupby("Type")["Store"].nunique().reset_index()
    axes[1].bar(
        type_counts["Type"],
        type_counts["Store"],
        color=[TYPE_COLORS[t] for t in type_counts["Type"]],
    )
    axes[1].set_title("Number of Stores by Type", fontweight="bold")
    axes[1].set_ylabel("Store Count")

    type_class = df.groupby("Type")["Sales_Class"].mean() * 100
    axes[2].bar(
        type_class.index,
        type_class.values,
        color=[TYPE_COLORS[t] for t in type_class.index],
    )
    axes[2].axhline(50, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_title("% High-Sales by Store Type", fontweight="bold")
    axes[2].set_ylabel("High Class %")

    fig.tight_layout()
    paths.append(_save_fig(fig, "03_store_type_comparison"))

    # ── 3b. Size vs Sales scatter ──
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    store_agg = (
        df.groupby(["Store", "Type"])
        .agg(
            Avg_Sales=("Weekly_Sales", "mean"),
            Size=("Size", "first"),
            High_Pct=("Sales_Class", "mean"),
        )
        .reset_index()
    )

    scatter = ax.scatter(
        store_agg["Size"],
        store_agg["Avg_Sales"],
        c=store_agg["High_Pct"],
        cmap="RdYlBu_r",
        s=100,
        edgecolors="white",
        linewidth=0.5,
        alpha=0.8,
    )
    plt.colorbar(scatter, ax=ax, label="% High Sales")
    for _, row in store_agg.iterrows():
        ax.annotate(
            str(int(row["Store"])),
            (row["Size"], row["Avg_Sales"]),
            fontsize=7,
            ha="center",
            va="bottom",
        )
    ax.set_title(
        "Store Size vs Average Sales (colored by High-Sales %)", fontweight="bold"
    )
    ax.set_xlabel("Store Size (sq ft)")
    ax.set_ylabel("Average Weekly Sales ($)")
    fig.tight_layout()
    paths.append(_save_fig(fig, "03_size_vs_sales_scatter"))

    # ── 3c. Top/Bottom stores ──
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    store_avg = df.groupby("Store")["Weekly_Sales"].mean().sort_values()
    top10 = store_avg.tail(10)
    bot10 = store_avg.head(10)

    axes[0].barh(top10.index.astype(str), top10.values, color=PALETTE["success"])
    axes[0].set_title("Top 10 Stores (Avg Sales)", fontweight="bold")
    axes[0].set_xlabel("Avg Weekly Sales ($)")

    axes[1].barh(bot10.index.astype(str), bot10.values, color=PALETTE["danger"])
    axes[1].set_title("Bottom 10 Stores (Avg Sales)", fontweight="bold")
    axes[1].set_xlabel("Avg Weekly Sales ($)")

    fig.tight_layout()
    paths.append(_save_fig(fig, "03_top_bottom_stores"))

    # ── 3d. Department analysis ──
    fig, ax = plt.subplots(figsize=(16, 6))
    dept_avg = (
        df.groupby("Dept")["Weekly_Sales"].mean().sort_values(ascending=False).head(20)
    )
    colors = [
        PALETTE["primary"] if v > df["Weekly_Sales"].mean() else PALETTE["light"]
        for v in dept_avg.values
    ]
    ax.bar(dept_avg.index.astype(str), dept_avg.values, color=colors, edgecolor="white")
    ax.axhline(
        df["Weekly_Sales"].mean(),
        color=PALETTE["danger"],
        linestyle="--",
        label=f"Overall Mean: ${df['Weekly_Sales'].mean():,.0f}",
    )
    ax.set_title("Top 20 Departments by Average Sales", fontweight="bold")
    ax.set_xlabel("Department")
    ax.set_ylabel("Avg Weekly Sales ($)")
    ax.legend()
    plt.xticks(rotation=45)
    fig.tight_layout()
    paths.append(_save_fig(fig, "03_department_analysis"))

    # ── 3e. Store performance heatmap ──
    fig, ax = plt.subplots(figsize=(14, 10))
    store_monthly = df.pivot_table(
        values="Weekly_Sales",
        index="Store",
        columns="Month",
        aggfunc="mean",
    )
    sns.heatmap(
        store_monthly,
        cmap="YlOrRd",
        ax=ax,
        fmt=".0f",
        cbar_kws={"label": "Avg Sales ($)"},
        yticklabels=True,
    )
    ax.set_title("Store × Month Performance Heatmap", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Store ID")
    fig.tight_layout()
    paths.append(_save_fig(fig, "03_store_month_heatmap"))

    report["store_analysis"] = {
        "stores_per_type": df.groupby("Type")["Store"].nunique().to_dict(),
        "avg_sales_by_type": df.groupby("Type")["Weekly_Sales"]
        .mean()
        .round(2)
        .to_dict(),
        "top_5_stores": store_avg.tail(5).round(2).to_dict(),
        "bottom_5_stores": store_avg.head(5).round(2).to_dict(),
    }

    return paths


# GROUP 4: FEATURE DISTRIBUTIONS
def plot_feature_distributions(df: pd.DataFrame, report: dict) -> list[Path]:
    """Distribution analysis of key continuous features."""
    logger.info("Group 4: Feature Distributions …")
    paths = []

    # ── 4a. Distribution grid ──
    plot_features = [c for c in CONTINUOUS_FEATURES if c in df.columns][:12]
    n_cols = 4
    n_rows = (len(plot_features) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(plot_features):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=50, color=PALETTE["primary"], alpha=0.7, edgecolor="white")
        ax.axvline(
            data.mean(),
            color=PALETTE["danger"],
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {data.mean():,.1f}",
        )
        ax.axvline(
            data.median(),
            color=PALETTE["warning"],
            linestyle="-",
            linewidth=1.5,
            label=f"Median: {data.median():,.1f}",
        )
        ax.set_title(f"{col} (skew={data.skew():.2f})", fontweight="bold", fontsize=10)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("FEATURE DISTRIBUTIONS", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    paths.append(_save_fig(fig, "04_distribution_grid"))

    # ── 4b. Distribution by class (key features) ──
    key_feats = [c for c in KEY_FEATURES if c in df.columns][:8]
    n_rows = (len(key_feats) + 3) // 4

    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(key_feats):
        ax = axes[i]
        for label, color in CLASS_COLORS.items():
            subset = df[df["Sales_Label"] == label][col].dropna()
            ax.hist(subset, bins=40, alpha=0.5, label=label, color=color, density=True)
        ax.set_title(f"{col} by Class", fontweight="bold", fontsize=10)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "FEATURE DISTRIBUTIONS BY TARGET CLASS", fontsize=16, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    paths.append(_save_fig(fig, "04_distribution_by_class"))

    # ── 4c. QQ plots for normality check ──
    qq_features = [
        c
        for c in ["Weekly_Sales", "Temperature", "CPI", "Unemployment"]
        if c in df.columns
    ]
    fig, axes = plt.subplots(1, len(qq_features), figsize=(5 * len(qq_features), 5))
    if len(qq_features) == 1:
        axes = [axes]

    for ax, col in zip(axes, qq_features):
        data = df[col].dropna().sample(min(5000, len(df)), random_state=42)
        scipy_stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(f"QQ Plot: {col}", fontweight="bold")

    fig.tight_layout()
    paths.append(_save_fig(fig, "04_qq_plots"))

    # ── 4d. Skewness summary ──
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    skew_data = df[plot_features].skew().sort_values()
    colors = [
        "#EF4444" if abs(v) > 2 else "#F59E0B" if abs(v) > 1 else "#10B981"
        for v in skew_data.values
    ]
    ax.barh(skew_data.index, skew_data.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(-1, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(1, color="gray", linestyle="--", alpha=0.3)
    ax.set_title(
        "Feature Skewness (|>2|=Red, |>1|=Yellow, else=Green)", fontweight="bold"
    )
    ax.set_xlabel("Skewness")
    fig.tight_layout()
    paths.append(_save_fig(fig, "04_skewness_summary"))

    report["distributions"] = {
        "skewness": df[plot_features].skew().round(4).to_dict(),
        "kurtosis": df[plot_features].kurtosis().round(4).to_dict(),
    }

    return paths


# GROUP 5: CORRELATION ANALYSIS
def plot_correlation_analysis(df: pd.DataFrame, report: dict) -> list[Path]:
    logger.info("Group 5: Correlation Analysis …")
    paths = []

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in ["Store", "Dept", "Sales_Class"]
    ]
    corr_cols = [c for c in numeric_cols if c in df.columns][:25]

    # ── 5a. Full correlation heatmap ──
    fig, ax = plt.subplots(figsize=(18, 14))
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        ax=ax,
        annot=False,
        square=True,
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    fig.tight_layout()
    paths.append(_save_fig(fig, "05_correlation_heatmap"))

    # ── 5b. Target correlation ranking ──
    fig, ax = plt.subplots(figsize=(12, 10))
    target_corr = (
        df[corr_cols + ["Sales_Class"]].corr()["Sales_Class"].drop("Sales_Class")
    )
    target_corr = target_corr.sort_values()
    colors = [
        PALETTE["danger"] if v < 0 else PALETTE["success"] for v in target_corr.values
    ]
    ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Feature Correlation with Target (Sales_Class)", fontweight="bold")
    ax.set_xlabel("Pearson Correlation")
    fig.tight_layout()
    paths.append(_save_fig(fig, "05_target_correlation_ranking"))

    # ── 5c. Top correlated pairs (multicollinearity check) ──
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Find top correlated pairs (excluding self-correlation)
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            pairs.append(
                {
                    "pair": f"{corr.columns[i]} ↔ {corr.columns[j]}",
                    "r": corr.iloc[i, j],
                }
            )
    pairs_df = pd.DataFrame(pairs)
    pairs_df["abs_r"] = pairs_df["r"].abs()
    top_pairs = pairs_df.nlargest(15, "abs_r")

    colors = [
        "#EF4444" if abs(r) > 0.7 else "#F59E0B" if abs(r) > 0.5 else "#10B981"
        for r in top_pairs["r"]
    ]
    ax.barh(top_pairs["pair"], top_pairs["r"], color=colors, edgecolor="white")
    ax.axvline(0.7, color="red", linestyle="--", alpha=0.3, label="|r| > 0.7 (high)")
    ax.axvline(-0.7, color="red", linestyle="--", alpha=0.3)
    ax.set_title("Top 15 Correlated Feature Pairs", fontweight="bold")
    ax.set_xlabel("Pearson r")
    ax.legend()
    fig.tight_layout()
    paths.append(_save_fig(fig, "05_top_correlated_pairs"))

    # ── 5d. Key feature scatter matrix ──
    scatter_features = [
        c
        for c in ["Size", "Temperature", "Unemployment", "TotalMarkDown", "EconIndex"]
        if c in df.columns
    ]
    if len(scatter_features) >= 3:
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig = sns.pairplot(
            sample,
            vars=scatter_features,
            hue="Sales_Label",
            palette=CLASS_COLORS,
            diag_kind="kde",
            plot_kws={"alpha": 0.3, "s": 10},
            height=2.5,
        )
        fig.fig.suptitle("Key Feature Pair Plot by Class", fontweight="bold", y=1.02)
        paths.append(_save_fig(fig.fig, "05_pairplot"))

    report["correlations"] = {
        "top_target_positive": target_corr.nlargest(5).round(4).to_dict(),
        "top_target_negative": target_corr.nsmallest(5).round(4).to_dict(),
        "high_collinearity_pairs": (
            top_pairs[top_pairs["abs_r"] > 0.7][["pair", "r"]].to_dict(orient="records")
        ),
    }

    return paths


# GROUP 6: PROMOTION IMPACT
def plot_promotion_impact(df: pd.DataFrame, report: dict) -> list[Path]:
    logger.info("Group 6: Promotion Impact …")
    paths = []

    # ── 6a. Promotion vs no promotion ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if "HasAnyMarkDown" in df.columns:
        promo_labels = {0: "No Promo", 1: "Has Promo"}
        df_plot = df.copy()
        df_plot["PromoLabel"] = df_plot["HasAnyMarkDown"].map(promo_labels)

        sns.boxplot(
            data=df_plot,
            x="PromoLabel",
            y="Weekly_Sales",
            ax=axes[0],
            palette=[PALETTE["light"], PALETTE["primary"]],
        )
        axes[0].set_title("Sales: Promo vs No Promo", fontweight="bold")
        axes[0].set_ylim(-5000, 60000)

        promo_class = df_plot.groupby("PromoLabel")["Sales_Class"].mean() * 100
        axes[1].bar(
            promo_class.index,
            promo_class.values,
            color=[PALETTE["light"], PALETTE["primary"]],
        )
        axes[1].axhline(50, color="gray", linestyle="--", alpha=0.5)
        axes[1].set_title("% High-Sales: Promo vs No Promo", fontweight="bold")
        axes[1].set_ylabel("High Class %")

    if "ActiveMarkDownCount" in df.columns:
        count_sales = df.groupby("ActiveMarkDownCount")["Weekly_Sales"].mean()
        axes[2].bar(
            count_sales.index,
            count_sales.values,
            color=PALETTE["primary"],
            edgecolor="white",
        )
        axes[2].set_title("Avg Sales by # Active MarkDowns", fontweight="bold")
        axes[2].set_xlabel("Number of Active MarkDowns")
        axes[2].set_ylabel("Avg Weekly Sales ($)")

    fig.tight_layout()
    paths.append(_save_fig(fig, "06_promotion_impact"))

    # ── 6b. MarkDown breakdown ──
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    md_cols = [c for c in MARKDOWN_COLS if c in df.columns]
    if md_cols:
        md_means = {}
        for col in md_cols:
            nonzero = df[df[col] > 0][col]
            md_means[col] = nonzero.mean() if len(nonzero) > 0 else 0

        axes[0].bar(
            md_means.keys(),
            md_means.values(),
            color=PALETTE["secondary"],
            edgecolor="white",
        )
        axes[0].set_title("Avg MarkDown Amount (when active)", fontweight="bold")
        axes[0].set_ylabel("Average Amount ($)")
        axes[0].tick_params(axis="x", rotation=45)

        md_corr = {col: df[col].corr(df["Weekly_Sales"]) for col in md_cols}
        axes[1].bar(
            md_corr.keys(),
            md_corr.values(),
            color=PALETTE["primary"],
            edgecolor="white",
        )
        axes[1].set_title("MarkDown Correlation with Weekly Sales", fontweight="bold")
        axes[1].set_ylabel("Pearson r")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].axhline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    paths.append(_save_fig(fig, "06_markdown_breakdown"))

    # ── 6c. Promotion effectiveness by store type ──
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    if "HasAnyMarkDown" in df.columns:
        promo_type = (
            df.groupby(["Type", "HasAnyMarkDown"])["Weekly_Sales"].mean().unstack()
        )
        promo_type.columns = ["No Promo", "Has Promo"]
        promo_type.plot(
            kind="bar",
            ax=ax,
            color=[PALETTE["light"], PALETTE["primary"]],
            edgecolor="white",
        )
        ax.set_title("Promotion Effect by Store Type", fontweight="bold")
        ax.set_ylabel("Avg Weekly Sales ($)")
        ax.set_xlabel("Store Type")
        ax.legend(title="Promotion")
        plt.xticks(rotation=0)
    fig.tight_layout()
    paths.append(_save_fig(fig, "06_promo_by_store_type"))

    # ── 6d. Total markdown vs sales ──
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    if "TotalMarkDown" in df.columns:
        promo_only = df[df["TotalMarkDown"] > 0].sample(
            min(5000, len(df)), random_state=42
        )
        scatter = ax.scatter(
            promo_only["TotalMarkDown"],
            promo_only["Weekly_Sales"],
            c=promo_only["Sales_Class"],
            cmap="RdYlBu_r",
            alpha=0.3,
            s=10,
            edgecolors="none",
        )
        plt.colorbar(scatter, ax=ax, label="Sales Class (0=Low, 1=High)")
        ax.set_title("Total MarkDown vs Weekly Sales", fontweight="bold")
        ax.set_xlabel("Total MarkDown ($)")
        ax.set_ylabel("Weekly Sales ($)")
    fig.tight_layout()
    paths.append(_save_fig(fig, "06_markdown_vs_sales"))

    report["promotions"] = {
        "promo_vs_no_promo_avg_sales": (
            df.groupby("HasAnyMarkDown")["Weekly_Sales"].mean().round(2).to_dict()
            if "HasAnyMarkDown" in df.columns
            else {}
        ),
    }

    return paths


# GROUP 7: ECONOMIC INDICATORS
def plot_economic_indicators(df: pd.DataFrame, report: dict) -> list[Path]:
    logger.info("Group 7: Economic Indicators …")
    paths = []

    # ── 7a. Economic trends over time ──
    econ_cols = [
        c
        for c in ["UMCSENT", "RSXFS", "PCE", "Unemployment", "CPI", "Fuel_Price"]
        if c in df.columns
    ]

    fig, axes = plt.subplots(
        len(econ_cols), 1, figsize=(16, 3 * len(econ_cols)), sharex=True
    )
    if len(econ_cols) == 1:
        axes = [axes]

    econ_ts = df.groupby("Date")[econ_cols].mean().reset_index()
    for ax, col in zip(axes, econ_cols):
        ax.plot(econ_ts["Date"], econ_ts[col], color=PALETTE["primary"], linewidth=1.5)
        ax.fill_between(
            econ_ts["Date"], econ_ts[col], alpha=0.1, color=PALETTE["primary"]
        )
        ax.set_ylabel(col, fontsize=10)
        ax.tick_params(labelsize=8)

    axes[0].set_title("Economic Indicators Over Time", fontweight="bold")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    paths.append(_save_fig(fig, "07_economic_trends"))

    # ── 7b. Economic features vs target ──
    econ_feats = [
        c
        for c in ["EconIndex", "ConsumerConfRatio", "FuelBurden", "PurchasingPower"]
        if c in df.columns
    ]

    if econ_feats:
        fig, axes = plt.subplots(1, len(econ_feats), figsize=(5 * len(econ_feats), 5))
        if len(econ_feats) == 1:
            axes = [axes]

        for ax, col in zip(axes, econ_feats):
            sns.boxplot(data=df, x="Sales_Label", y=col, palette=CLASS_COLORS, ax=ax)
            ax.set_title(f"{col} by Class", fontweight="bold", fontsize=10)

        fig.suptitle(
            "Economic Features by Target Class", fontsize=14, fontweight="bold", y=1.02
        )
        fig.tight_layout()
        paths.append(_save_fig(fig, "07_economic_vs_target"))

    # ── 7c. FRED multicollinearity ──
    fred_cols = [c for c in ["UMCSENT", "RSXFS", "PCE"] if c in df.columns]
    if len(fred_cols) >= 2:
        fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
        fred_corr = df[fred_cols].corr()
        sns.heatmap(
            fred_corr,
            annot=True,
            cmap="RdBu_r",
            center=0,
            ax=ax,
            square=True,
            fmt=".3f",
            vmin=-1,
            vmax=1,
        )
        ax.set_title("FRED Series Multicollinearity", fontweight="bold")
        fig.tight_layout()
        paths.append(_save_fig(fig, "07_fred_multicollinearity"))

    # ── 7d. Unemployment vs sales by store type ──
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    if "Unemployment" in df.columns:
        for type_val, color in TYPE_COLORS.items():
            sub = df[df["Type"] == type_val].sample(
                min(2000, len(df[df["Type"] == type_val])), random_state=42
            )
            ax.scatter(
                sub["Unemployment"],
                sub["Weekly_Sales"],
                alpha=0.2,
                s=8,
                color=color,
                label=f"Type {type_val}",
            )
        ax.set_title("Unemployment vs Sales by Store Type", fontweight="bold")
        ax.set_xlabel("Unemployment Rate (%)")
        ax.set_ylabel("Weekly Sales ($)")
        ax.set_ylim(-5000, 80000)
        ax.legend()
    fig.tight_layout()
    paths.append(_save_fig(fig, "07_unemployment_vs_sales"))

    return paths


# GROUP 8: FEATURE IMPORTANCE (PRELIMINARY)
def plot_feature_importance(df: pd.DataFrame, report: dict) -> list[Path]:
    logger.info("Group 8: Feature Importance (Preliminary) …")
    paths = []

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in ["Sales_Class", "Store", "Dept"]
    ]

    # ── 8a. Mutual information / point-biserial correlation ──
    importance = {}
    for col in numeric_cols:
        clean = df[[col, "Sales_Class"]].dropna()
        if len(clean) < 100:
            continue
        r, p = scipy_stats.pointbiserialr(clean["Sales_Class"], clean[col])
        importance[col] = {
            "correlation": abs(r),
            "p_value": p,
            "direction": "+" if r > 0 else "-",
        }

    imp_df = pd.DataFrame(importance).T.sort_values("correlation", ascending=True)
    imp_df = imp_df.tail(25)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = [
        PALETTE["success"] if d == "+" else PALETTE["danger"]
        for d in imp_df["direction"]
    ]
    ax.barh(imp_df.index, imp_df["correlation"], color=colors, edgecolor="white")
    ax.set_title(
        "Top 25 Features by |Point-Biserial Correlation| with Target\n"
        "(Green=Positive, Red=Negative)",
        fontweight="bold",
    )
    ax.set_xlabel("|Correlation|")
    fig.tight_layout()
    paths.append(_save_fig(fig, "08_feature_importance_correlation"))

    # ── 8b. T-test significance ──
    ttest_results = {}
    for col in numeric_cols:
        low = df[df["Sales_Class"] == 0][col].dropna()
        high = df[df["Sales_Class"] == 1][col].dropna()
        if len(low) < 30 or len(high) < 30:
            continue
        stat, p = scipy_stats.ttest_ind(low, high, equal_var=False)
        ttest_results[col] = {"t_stat": stat, "p_value": p, "significant": p < 0.05}

    sig_df = pd.DataFrame(ttest_results).T
    pvals = pd.to_numeric(sig_df["p_value"], errors="coerce").clip(lower=1e-300)
    sig_df["neg_log_p"] = -np.log10(pvals)
    sig_df = sig_df.sort_values("neg_log_p", ascending=True).tail(25)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = [
        PALETTE["danger"] if s else PALETTE["light"] for s in sig_df["significant"]
    ]
    ax.barh(sig_df.index, sig_df["neg_log_p"], color=colors, edgecolor="white")
    ax.axvline(-np.log10(0.05), color="black", linestyle="--", label="p=0.05 threshold")
    ax.set_title(
        "Top 25 Features by Statistical Significance (T-test)", fontweight="bold"
    )
    ax.set_xlabel("-log₁₀(p-value)")
    ax.legend()
    fig.tight_layout()
    paths.append(_save_fig(fig, "08_feature_significance_ttest"))

    # ── 8c. Effect size (Cohen's d) ──
    cohens_d = {}
    for col in numeric_cols:
        low = df[df["Sales_Class"] == 0][col].dropna()
        high = df[df["Sales_Class"] == 1][col].dropna()
        if len(low) < 30 or len(high) < 30:
            continue
        pooled_std = np.sqrt((low.std() ** 2 + high.std() ** 2) / 2)
        if pooled_std > 0:
            d = (high.mean() - low.mean()) / pooled_std
            cohens_d[col] = abs(d)

    cd_series = pd.Series(cohens_d).sort_values(ascending=True).tail(25)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = [
        "#EF4444" if v > 0.8 else "#F59E0B" if v > 0.5 else "#10B981"
        for v in cd_series.values
    ]
    ax.barh(cd_series.index, cd_series.values, color=colors, edgecolor="white")
    ax.axvline(0.2, color="gray", linestyle="--", alpha=0.3, label="Small (0.2)")
    ax.axvline(0.5, color="gray", linestyle="-.", alpha=0.3, label="Medium (0.5)")
    ax.axvline(0.8, color="gray", linestyle=":", alpha=0.3, label="Large (0.8)")
    ax.set_title("Top 25 Features by Effect Size (|Cohen's d|)", fontweight="bold")
    ax.set_xlabel("|Cohen's d|")
    ax.legend()
    fig.tight_layout()
    paths.append(_save_fig(fig, "08_effect_size_cohens_d"))

    report["feature_importance"] = {
        "top_10_correlation": pd.to_numeric(
            imp_df.tail(10)["correlation"], errors="coerce"
        )
        .round(4)
        .to_dict(),
        "top_10_cohens_d": cd_series.tail(10).round(4).to_dict(),
        "significant_features_count": sum(
            1 for v in ttest_results.values() if v["significant"]
        ),
        "total_features_tested": len(ttest_results),
    }

    return paths


# GROUP 9: SEGMENTATION ANALYSIS
def plot_segmentation(df: pd.DataFrame, report: dict) -> list[Path]:
    logger.info("Group 9: Segmentation Analysis …")
    paths = []

    # ── 9a. Type × Holiday × Class ──
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    seg = df.groupby(["Type", "IsHoliday"])["Sales_Class"].mean().unstack() * 100
    seg.columns = ["Non-Holiday", "Holiday"]
    seg.plot(
        kind="bar",
        ax=ax,
        color=[PALETTE["primary"], PALETTE["danger"]],
        edgecolor="white",
    )
    ax.set_title("% High-Sales by Store Type & Holiday", fontweight="bold")
    ax.set_ylabel("High Class %")
    ax.set_xlabel("Store Type")
    ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
    ax.legend(title="Period")
    plt.xticks(rotation=0)
    fig.tight_layout()
    paths.append(_save_fig(fig, "09_type_holiday_segmentation"))

    # ── 9b. Size quartile × Season ──
    if "SizeQuartile" in df.columns:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        seg2 = (
            df.groupby(["SizeQuartile", "Quarter"])["Sales_Class"].mean().unstack()
            * 100
        )
        seg2.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
        ax.set_title("% High-Sales by Store Size Quartile & Quarter", fontweight="bold")
        ax.set_ylabel("High Class %")
        ax.set_xlabel("Size Quartile (1=Small → 4=Large)")
        ax.axhline(50, color="gray", linestyle="--", alpha=0.5)
        ax.legend(title="Quarter")
        plt.xticks(rotation=0)
        fig.tight_layout()
        paths.append(_save_fig(fig, "09_size_quarter_segmentation"))

    # ── 9c. Promotion × Holiday × Sales ──
    if "HasAnyMarkDown" in df.columns:
        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        seg3 = (
            df.groupby(["HasAnyMarkDown", "IsHoliday"])["Weekly_Sales"].mean().unstack()
        )
        seg3.index = ["No Promo", "Has Promo"]
        seg3.columns = ["Non-Holiday", "Holiday"]
        seg3.plot(
            kind="bar",
            ax=ax,
            color=[PALETTE["primary"], PALETTE["danger"]],
            edgecolor="white",
        )
        ax.set_title("Avg Sales: Promotion × Holiday Interaction", fontweight="bold")
        ax.set_ylabel("Average Weekly Sales ($)")
        ax.legend(title="Period")
        plt.xticks(rotation=0)
        fig.tight_layout()
        paths.append(_save_fig(fig, "09_promo_holiday_interaction"))

    # ── 9d. Multi-segment summary heatmap ──
    fig, ax = plt.subplots(figsize=(12, 8))
    if "SizeQuartile" in df.columns:
        seg4 = (
            df.pivot_table(
                values="Sales_Class",
                index="Type",
                columns="SizeQuartile",
                aggfunc="mean",
            )
            * 100
        )
        sns.heatmap(
            seg4,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            ax=ax,
            center=50,
            cbar_kws={"label": "% High Sales"},
            linewidths=1,
            linecolor="white",
        )
        ax.set_title("% High-Sales: Store Type × Size Quartile", fontweight="bold")
        ax.set_xlabel("Size Quartile (1=Small → 4=Large)")
        ax.set_ylabel("Store Type")
    fig.tight_layout()
    paths.append(_save_fig(fig, "09_type_size_heatmap"))

    return paths


# DASHBOARD DATA EXPORT
def export_dashboard_data(df: pd.DataFrame, report: dict) -> None:
    logger.info("Exporting dashboard data …")

    dashboard = {}

    dashboard["weekly_sales_ts"] = (
        df.groupby("Date")
        .agg(
            avg_sales=("Weekly_Sales", "mean"),
            total_sales=("Weekly_Sales", "sum"),
            high_pct=("Sales_Class", "mean"),
            count=("Weekly_Sales", "count"),
        )
        .reset_index()
        .to_dict(orient="records")
    )

    dashboard["monthly_summary"] = (
        df.groupby("Month")
        .agg(
            avg_sales=("Weekly_Sales", "mean"),
            high_pct=("Sales_Class", "mean"),
            avg_temp=("Temperature", "mean"),
            avg_unemployment=("Unemployment", "mean"),
        )
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    dashboard["store_summary"] = (
        df.groupby(["Store", "Type"])
        .agg(
            avg_sales=("Weekly_Sales", "mean"),
            size=("Size", "first"),
            high_pct=("Sales_Class", "mean"),
            dept_count=("Dept", "nunique"),
        )
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    dashboard["dept_summary"] = (
        df.groupby("Dept")
        .agg(
            avg_sales=("Weekly_Sales", "mean"),
            high_pct=("Sales_Class", "mean"),
            store_count=("Store", "nunique"),
        )
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    if "HolidayName" in df.columns:
        dashboard["holiday_summary"] = (
            df.groupby("HolidayName")
            .agg(
                avg_sales=("Weekly_Sales", "mean"),
                high_pct=("Sales_Class", "mean"),
                count=("Weekly_Sales", "count"),
            )
            .round(2)
            .reset_index()
            .to_dict(orient="records")
        )

    if "HasAnyMarkDown" in df.columns:
        dashboard["promotion_summary"] = (
            df.groupby("HasAnyMarkDown")
            .agg(
                avg_sales=("Weekly_Sales", "mean"),
                high_pct=("Sales_Class", "mean"),
                count=("Weekly_Sales", "count"),
            )
            .round(2)
            .reset_index()
            .to_dict(orient="records")
        )

    dist = df["Sales_Label"].value_counts()
    dashboard["class_distribution"] = {
        "counts": dist.to_dict(),
        "percentages": (dist / len(df) * 100).round(2).to_dict(),
    }

    dashboard["kpi_metrics"] = {
        "total_records": len(df),
        "total_stores": int(df["Store"].nunique()),
        "total_departments": int(df["Dept"].nunique()),
        "avg_weekly_sales": round(float(df["Weekly_Sales"].mean()), 2),
        "median_weekly_sales": round(float(df["Weekly_Sales"].median()), 2),
        "high_sales_pct": round(float(df["Sales_Class"].mean() * 100), 2),
        "date_range": {
            "start": str(df["Date"].min().date()),
            "end": str(df["Date"].max()),
        },
        "avg_temperature": round(float(df["Temperature"].mean()), 2),
        "avg_unemployment": round(float(df["Unemployment"].mean()), 2),
        "avg_fuel_price": round(float(df["Fuel_Price"].mean()), 2),
        "promo_pct": (
            round(float(df["HasAnyMarkDown"].mean() * 100), 2)
            if "HasAnyMarkDown" in df.columns
            else 0.0
        ),
    }

    feature_stats = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ["Store", "Dept"]:
            continue
        s = df[col].dropna()
        feature_stats[col] = {
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "skewness": round(float(s.skew()), 4),
        }
    dashboard["feature_statistics"] = feature_stats

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in ["Sales_Class", "Store", "Dept"]
    ]
    target_corr = (
        df[numeric_cols + ["Sales_Class"]].corr()["Sales_Class"].drop("Sales_Class")
    )
    dashboard["target_correlations"] = {
        k: round(float(v), 4)
        for k, v in target_corr.sort_values(ascending=False).items()
    }

    dashboard["store_type_breakdown"] = (
        df.groupby("Type")
        .agg(
            store_count=("Store", "nunique"),
            avg_sales=("Weekly_Sales", "mean"),
            avg_size=("Size", "mean"),
            high_pct=("Sales_Class", "mean"),
        )
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    dashboard["year_summary"] = (
        df.groupby("Year")
        .agg(
            avg_sales=("Weekly_Sales", "mean"),
            total_sales=("Weekly_Sales", "sum"),
            high_pct=("Sales_Class", "mean"),
            weeks=("Date", "nunique"),
        )
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    with open(DASHBOARD_DATA_PATH, "w") as f:
        json.dump(dashboard, f, indent=2, default=_safe_json)

    logger.info(
        "Dashboard data saved to: {} ({} sections)", DASHBOARD_DATA_PATH, len(dashboard)
    )
    report["dashboard_data"] = {
        "path": str(DASHBOARD_DATA_PATH),
        "sections": list(dashboard.keys()),
        "section_count": len(dashboard),
    }


# SUMMARY STATISTICS
def compute_summary_statistics(df: pd.DataFrame, report: dict) -> dict:
    logger.info("Computing summary statistics …")

    summary = {
        "dataset": {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "date_range": {
                "start": (
                    str(df["Date"].min().date()) if "Date" in df.columns else "N/A"
                ),
                "end": str(df["Date"].max().date()) if "Date" in df.columns else "N/A",
            },
        },
        "target": {
            "class_counts": (
                df["Sales_Label"].value_counts().to_dict()
                if "Sales_Label" in df.columns
                else {}
            ),
            "balance_ratio": round(
                df["Sales_Class"].value_counts().max()
                / df["Sales_Class"].value_counts().min(),
                4,
            ),
        },
        "numeric_summary": {},
        "categorical_summary": {},
        "missing_summary": {},
    }

    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ["Store", "Dept"]:
            continue
        s = df[col].dropna()
        summary["numeric_summary"][col] = {
            "count": int(len(s)),
            "null_pct": round(float(df[col].isna().mean() * 100), 2),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "p25": round(float(s.quantile(0.25)), 4),
            "median": round(float(s.median()), 4),
            "p75": round(float(s.quantile(0.75)), 4),
            "max": round(float(s.max()), 4),
            "skewness": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
        }

    for col in ["Type", "Sales_Label", "StoreTypeLabel", "HolidayName", "SalesBucket"]:
        if col in df.columns:
            vc = df[col].value_counts()
            summary["categorical_summary"][col] = {
                "unique": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in vc.head(10).items()},
                "dominant_pct": round(float(vc.iloc[0] / len(df) * 100), 2),
            }

    missing = df.isna().sum()
    if missing.sum() > 0:
        summary["missing_summary"] = {
            col: {"count": int(v), "pct": round(float(v / len(df) * 100), 2)}
            for col, v in missing[missing > 0].items()
        }
    else:
        summary["missing_summary"] = {"status": "No missing values"}

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=_safe_json)

    logger.info("Summary statistics saved to: {}", SUMMARY_PATH)
    report["summary_statistics"] = summary
    return summary


# ORCHESTRATOR
def run_eda(df: pd.DataFrame) -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("Starting EDA Pipeline")
    logger.info("=" * 60)
    logger.info("Input shape: {}", _shape_str(df))

    _setup_style()
    for d in [EDA_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"figures": []}
    all_paths = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        all_paths.extend(plot_target_analysis(df, report))
        all_paths.extend(plot_temporal_patterns(df, report))
        all_paths.extend(plot_store_analysis(df, report))
        all_paths.extend(plot_feature_distributions(df, report))
        all_paths.extend(plot_correlation_analysis(df, report))
        all_paths.extend(plot_promotion_impact(df, report))
        all_paths.extend(plot_economic_indicators(df, report))
        all_paths.extend(plot_feature_importance(df, report))
        all_paths.extend(plot_segmentation(df, report))

    export_dashboard_data(df, report)

    summary = compute_summary_statistics(df, report)

    report["figures"] = [str(p) for p in all_paths]
    report["figure_count"] = len(all_paths)

    eda_report_path = EDA_DIR / "eda_report.json"
    with open(eda_report_path, "w") as f:
        json.dump(report, f, indent=2, default=_safe_json)

    logger.info("")
    logger.info("=" * 60)
    logger.info("EDA SUMMARY")
    logger.info("=" * 60)
    logger.info("  Dataset:        {}", _shape_str(df))
    logger.info("  Figures saved:  {} → {}", len(all_paths), FIGURES_DIR)
    logger.info("  Dashboard data: {}", DASHBOARD_DATA_PATH)
    logger.info("  Summary stats:  {}", SUMMARY_PATH)
    logger.info("  EDA report:     {}", eda_report_path)
    logger.info("")

    groups = [
        ("Target Analysis", 4),
        ("Temporal Patterns", 6),
        ("Store Analysis", 5),
        ("Feature Distributions", 4),
        ("Correlation Analysis", 4),
        ("Promotion Impact", 4),
        ("Economic Indicators", 4),
        ("Feature Importance", 3),
        ("Segmentation", 4),
    ]
    for name, count in groups:
        logger.info("    {:30s}  {} plots", name, count)

    logger.info("")
    logger.info("  Total: {} visualizations", len(all_paths))
    logger.info("=" * 60)

    return {
        "report": report,
        "summary": summary,
        "figures_dir": str(FIGURES_DIR),
        "dashboard_data_path": str(DASHBOARD_DATA_PATH),
        "figure_paths": [str(p) for p in all_paths],
    }


# Entry point
if __name__ == "__main__":
    EDA_DATASET_PATH = Path("data/eda_ready/eda_dataset.csv")

    logger.info("Loading EDA dataset from: {}", EDA_DATASET_PATH)
    eda_df = pd.read_csv(EDA_DATASET_PATH, parse_dates=["Date"])

    results = run_eda(eda_df)
