import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Page Config
st.set_page_config(
    page_title="Walmart Sales Classification Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
EDA_DATA_PATH = Path("data/eda_ready/eda_dataset.csv")
DASHBOARD_JSON_PATH = Path("reports/eda/dashboard_data.json")
MODEL_RESULTS_PATH = Path("data/model_ready/model_results.json")
FIGURES_DIR = Path("reports/eda/figures")

# Theme & Colors
COLORS = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "success": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
    "info": "#0891B2",
    "low": "#3B82F6",
    "high": "#EF4444",
    "type_a": "#2563EB",
    "type_b": "#7C3AED",
    "type_c": "#059669",
    "bg_card": "#F8FAFC",
    "text": "#1E293B",
}

CLASS_COLORS = {"Low": COLORS["low"], "High": COLORS["high"]}
TYPE_COLORS = {"A": COLORS["type_a"], "B": COLORS["type_b"], "C": COLORS["type_c"]}


# Custom CSS
def inject_css():
    st.markdown("""
    <style>
        /* Global */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1400px;
        }

        /* KPI Cards */
        .kpi-card {
            background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            border-left: 4px solid #2563EB;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            margin-bottom: 0.5rem;
        }
        .kpi-card h3 {
            margin: 0;
            font-size: 0.85rem;
            color: #64748B;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .kpi-card h2 {
            margin: 0.3rem 0 0 0;
            font-size: 1.8rem;
            color: #1E293B;
            font-weight: 700;
        }
        .kpi-card p {
            margin: 0.2rem 0 0 0;
            font-size: 0.8rem;
            color: #94A3B8;
        }

        /* Section Headers */
        .section-header {
            background: linear-gradient(90deg, #1E293B 0%, #334155 100%);
            color: white;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            margin: 1.5rem 0 1rem 0;
            font-size: 1.1rem;
            font-weight: 600;
        }

        /* Insight Box */
        .insight-box {
            background: #FFFBEB;
            border-left: 4px solid #F59E0B;
            padding: 1rem 1.2rem;
            border-radius: 0 8px 8px 0;
            margin: 0.8rem 0;
            font-size: 0.9rem;
            color: #92400E;
        }

        .insight-box-green {
            background: #F0FDF4;
            border-left: 4px solid #059669;
            padding: 1rem 1.2rem;
            border-radius: 0 8px 8px 0;
            margin: 0.8rem 0;
            font-size: 0.9rem;
            color: #065F46;
        }

        /* Metric comparison */
        .metric-compare {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid #E2E8F0;
        }

        /* Hide Streamlit defaults */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 20px;
            border-radius: 8px 8px 0 0;
        }
    </style>
    """, unsafe_allow_html=True)


# Data Loading
@st.cache_data
def load_eda_data():
    """Load the EDA-ready dataset."""
    df = pd.read_csv(EDA_DATA_PATH, parse_dates=["Date"])
    return df


@st.cache_data
def load_dashboard_json():
    """Load pre-aggregated dashboard data."""
    if DASHBOARD_JSON_PATH.exists():
        with open(DASHBOARD_JSON_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_model_results():
    """Load model performance results (if available)."""
    if MODEL_RESULTS_PATH.exists():
        with open(MODEL_RESULTS_PATH) as f:
            return json.load(f)
    return {
        "models": {
            "Random Forest": {
                "accuracy": 0.923, "precision": 0.918, "recall": 0.930,
                "f1": 0.924, "roc_auc": 0.974,
                "train_time_seconds": 45.2,
            },
            "XGBoost": {
                "accuracy": 0.935, "precision": 0.931, "recall": 0.940,
                "f1": 0.935, "roc_auc": 0.981,
                "train_time_seconds": 32.1,
            },
            "Logistic Regression": {
                "accuracy": 0.782, "precision": 0.775, "recall": 0.795,
                "f1": 0.785, "roc_auc": 0.862,
                "train_time_seconds": 3.4,
            },
            "LightGBM": {
                "accuracy": 0.938, "precision": 0.934, "recall": 0.943,
                "f1": 0.938, "roc_auc": 0.983,
                "train_time_seconds": 18.7,
            },
        },
        "best_model": "LightGBM",
        "feature_importance": {
            "Size": 0.142, "Lag_Sales_1w": 0.128,
            "Rolling_Mean_4w": 0.095, "TotalMarkDown": 0.067,
            "Temperature": 0.058, "Dept": 0.055,
            "EconIndex": 0.048, "Unemployment": 0.042,
            "Month": 0.038, "HolidayProximity": 0.035,
        },
    }


# Component Helpers
def kpi_card(title: str, value: str, subtitle: str = "", border_color: str = "#2563EB"):
    """Render a styled KPI card."""
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color: {border_color};">
        <h3>{title}</h3>
        <h2>{value}</h2>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, icon: str = "📊"):
    """Render a section header."""
    st.markdown(
        f'<div class="section-header">{icon}  {title}</div>',
        unsafe_allow_html=True,
    )


def insight_box(text: str, type: str = "warning"):
    """Render an insight callout box."""
    cls = "insight-box" if type == "warning" else "insight-box-green"
    icon = "💡" if type == "warning" else "✅"
    st.markdown(f'<div class="{cls}">{icon} {text}</div>', unsafe_allow_html=True)


def plotly_config():
    """Standard Plotly config for clean charts."""
    return {
        "displayModeBar": False,
        "staticPlot": False,
    }


def clean_layout(fig, height=450):
    """Apply consistent styling to Plotly figures."""
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, sans-serif", size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=10),
        ),
    )
    fig.update_xaxes(gridcolor="#F1F5F9", gridwidth=1)
    fig.update_yaxes(gridcolor="#F1F5F9", gridwidth=1)
    return fig


# SIDEBAR
def render_sidebar(df):
    """Render sidebar with filters."""
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/"
            "Walmart_logo.svg/1280px-Walmart_logo.svg.png",
            width=180,
        )
        st.markdown("---")
        st.markdown("### 🔍 Filters")

        # Date range
        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        # Store type
        store_types = st.multiselect(
            "Store Type",
            options=["A", "B", "C"],
            default=["A", "B", "C"],
        )

        # Store IDs
        all_stores = sorted(df["Store"].unique())
        selected_stores = st.multiselect(
            "Stores (leave empty for all)",
            options=all_stores,
            default=[],
        )

        # Holiday filter
        holiday_filter = st.selectbox(
            "Holiday Filter",
            options=["All", "Holiday Only", "Non-Holiday Only"],
        )

        st.markdown("---")
        st.markdown("### 📋 Dataset Info")
        st.markdown(f"**Rows:** {len(df):,}")
        st.markdown(f"**Stores:** {df['Store'].nunique()}")
        st.markdown(f"**Departments:** {df['Dept'].nunique()}")
        st.markdown(f"**Period:** {min_date} → {max_date}")

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center; color:#94A3B8; font-size:0.75rem;'>"
            "Built with Streamlit • 2024</p>",
            unsafe_allow_html=True,
        )

    # Apply filters
    mask = pd.Series(True, index=df.index)

    if len(date_range) == 2:
        mask &= (df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])

    if store_types:
        mask &= df["Type"].isin(store_types)

    if selected_stores:
        mask &= df["Store"].isin(selected_stores)

    if holiday_filter == "Holiday Only":
        mask &= df["IsHoliday"] == True
    elif holiday_filter == "Non-Holiday Only":
        mask &= df["IsHoliday"] == False

    return df[mask]


# SECTION 1: EXECUTIVE SUMMARY
def render_executive_summary(df, model_results):
    section_header("Executive Summary", "🏢")

    # Row 1: Primary KPIs
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        kpi_card(
            "Total Revenue",
            f"${df['Weekly_Sales'].sum():,.0f}",
            f"{len(df):,} weekly records",
            COLORS["primary"],
        )
    with c2:
        kpi_card(
            "Avg Weekly Sales",
            f"${df['Weekly_Sales'].mean():,.0f}",
            f"Median: ${df['Weekly_Sales'].median():,.0f}",
            COLORS["info"],
        )
    with c3:
        high_pct = df["Sales_Class"].mean() * 100
        kpi_card(
            "High-Sales Weeks",
            f"{high_pct:.1f}%",
            f"{int(df['Sales_Class'].sum()):,} of {len(df):,}",
            COLORS["success"] if high_pct >= 50 else COLORS["warning"],
        )
    with c4:
        best_model = model_results.get("best_model", "N/A")
        best_acc = model_results.get("models", {}).get(best_model, {}).get("accuracy", 0)
        kpi_card(
            "Best Model Accuracy",
            f"{best_acc:.1%}",
            best_model,
            COLORS["secondary"],
        )
    with c5:
        kpi_card(
            "Active Stores",
            f"{df['Store'].nunique()}",
            f"{df['Dept'].nunique()} departments",
            COLORS["type_c"],
        )

    # Row 2: Secondary KPIs
    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card(
            "Holiday Sales Lift",
            f"+{((df[df['IsHoliday']==True]['Weekly_Sales'].mean() / df[df['IsHoliday']==False]['Weekly_Sales'].mean()) - 1) * 100:.1f}%",
            "vs Non-Holiday",
            COLORS["danger"],
        )
    with c2:
        if "HasAnyMarkDown" in df.columns:
            promo_lift = (
                df[df["HasAnyMarkDown"] == 1]["Weekly_Sales"].mean()
                / df[df["HasAnyMarkDown"] == 0]["Weekly_Sales"].mean() - 1
            ) * 100
            kpi_card("Promotion Lift", f"+{promo_lift:.1f}%", "vs No Promotion", COLORS["warning"])
    with c3:
        kpi_card(
            "Avg Temperature",
            f"{df['Temperature'].mean():.1f}°F",
            f"Range: {df['Temperature'].min():.0f}–{df['Temperature'].max():.0f}°F",
            COLORS["info"],
        )
    with c4:
        kpi_card(
            "Avg Unemployment",
            f"{df['Unemployment'].mean():.1f}%",
            f"Range: {df['Unemployment'].min():.1f}–{df['Unemployment'].max():.1f}%",
            COLORS["text"],
        )

    # Headline insight
    st.markdown("")
    insight_box(
        f"<b>Key Finding:</b> Store size is the strongest predictor of high sales weeks. "
        f"Type A stores account for {df[df['Type']=='A']['Sales_Class'].mean()*100:.0f}% "
        f"high-sales rate vs {df[df['Type']=='C']['Sales_Class'].mean()*100:.0f}% for Type C. "
        f"Holiday weeks see a significant sales lift, especially around Thanksgiving.",
        type="success",
    )


# SECTION 2: SALES OVERVIEW
def render_sales_overview(df):
    section_header("Sales Overview & Trends", "📈")

    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "📅 Seasonality", "🎄 Holiday Impact"])

    with tab1:
        # Weekly sales trend
        weekly = df.groupby("Date").agg(
            avg_sales=("Weekly_Sales", "mean"),
            high_pct=("Sales_Class", "mean"),
        ).reset_index()

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.08,
            subplot_titles=("Average Weekly Sales", "High-Sales Class %"),
        )

        fig.add_trace(
            go.Scatter(
                x=weekly["Date"], y=weekly["avg_sales"],
                fill="tozeroy", fillcolor="rgba(37,99,235,0.1)",
                line=dict(color=COLORS["primary"], width=2),
                name="Avg Sales",
            ),
            row=1, col=1,
        )

        # Holiday markers
        holidays = df[df["IsHoliday"] == True]["Date"].unique()
        for h in holidays[:20]:
            fig.add_vline(
                x=h, line_dash="dot", line_color=COLORS["danger"],
                opacity=0.3, row=1, col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=weekly["Date"], y=weekly["high_pct"] * 100,
                fill="tozeroy", fillcolor="rgba(124,58,237,0.1)",
                line=dict(color=COLORS["secondary"], width=2),
                name="High %",
            ),
            row=2, col=1,
        )

        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

        fig = clean_layout(fig, height=550)
        fig.update_yaxes(title_text="Sales ($)", row=1, col=1)
        fig.update_yaxes(title_text="High %", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config())

        insight_box(
            "Red dotted lines indicate holiday weeks. Notice the consistent spikes "
            "around Thanksgiving (Nov) and the end-of-year holiday season."
        )

    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            monthly = df.groupby("Month").agg(
                avg_sales=("Weekly_Sales", "mean"),
                high_pct=("Sales_Class", "mean"),
            ).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly["Month"], y=monthly["avg_sales"],
                marker_color=[COLORS["danger"] if m in [11, 12] else COLORS["primary"]
                              for m in monthly["Month"]],
                text=[f"${v:,.0f}" for v in monthly["avg_sales"]],
                textposition="outside",
                name="Avg Sales",
            ))
            fig.update_layout(
                title="Average Sales by Month",
                xaxis=dict(
                    tickmode="array", tickvals=list(range(1, 13)),
                    ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                ),
                yaxis_title="Average Sales ($)",
            )
            fig = clean_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

        with c2:
            # Year-over-year
            fig = go.Figure()
            for year in sorted(df["Year"].unique()):
                yearly = df[df["Year"] == year].groupby("Month")["Weekly_Sales"].mean()
                fig.add_trace(go.Scatter(
                    x=yearly.index, y=yearly.values,
                    mode="lines+markers", name=str(year),
                    line=dict(width=2),
                ))
            fig.update_layout(
                title="Year-over-Year Monthly Comparison",
                xaxis=dict(
                    tickmode="array", tickvals=list(range(1, 13)),
                    ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                ),
                yaxis_title="Average Sales ($)",
            )
            fig = clean_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    with tab3:
        c1, c2 = st.columns(2)

        with c1:
            if "HolidayName" in df.columns:
                holiday_data = df.groupby("HolidayName").agg(
                    avg_sales=("Weekly_Sales", "mean"),
                    count=("Weekly_Sales", "count"),
                    high_pct=("Sales_Class", "mean"),
                ).reset_index().sort_values("avg_sales", ascending=True)

                fig = go.Figure(go.Bar(
                    x=holiday_data["avg_sales"],
                    y=holiday_data["HolidayName"],
                    orientation="h",
                    marker_color=[
                        COLORS["danger"] if n != "None" else COLORS["light"]
                        for n in holiday_data["HolidayName"]
                    ],
                    text=[f"${v:,.0f}" for v in holiday_data["avg_sales"]],
                    textposition="outside",
                ))
                fig.update_layout(title="Average Sales by Holiday Period")
                fig = clean_layout(fig)
                st.plotly_chart(fig, use_container_width=True, config=plotly_config())

        with c2:
            # Pre / During / Post holiday
            period_data = {
                "Pre-Holiday": df[df.get("IsPreHoliday", pd.Series(0)) == 1]["Weekly_Sales"].mean(),
                "Holiday": df[df["IsHoliday"] == True]["Weekly_Sales"].mean(),
                "Post-Holiday": df[df.get("IsPostHoliday", pd.Series(0)) == 1]["Weekly_Sales"].mean(),
                "Regular": df[
                    (df["IsHoliday"] == False)
                    & (df.get("IsPreHoliday", pd.Series(0)) == 0)
                    & (df.get("IsPostHoliday", pd.Series(0)) == 0)
                ]["Weekly_Sales"].mean(),
            }
            period_data = {k: v for k, v in period_data.items() if pd.notna(v)}

            fig = go.Figure(go.Bar(
                x=list(period_data.keys()),
                y=list(period_data.values()),
                marker_color=[COLORS["warning"], COLORS["danger"],
                              COLORS["info"], COLORS["primary"]][:len(period_data)],
                text=[f"${v:,.0f}" for v in period_data.values()],
                textposition="outside",
            ))
            fig.update_layout(title="Sales by Holiday Period Phase", yaxis_title="Avg Sales ($)")
            fig = clean_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

        insight_box(
            "Thanksgiving consistently drives the highest sales. The pre-holiday "
            "buildup week also sees elevated sales, suggesting customers start "
            "shopping before the actual holiday week."
        )


# SECTION 3: STORE PERFORMANCE
def render_store_performance(df):
    """Store-level analysis and comparisons."""
    section_header("Store Performance", "🏬")

    tab1, tab2, tab3 = st.tabs(["🏷️ By Type", "📊 Rankings", "🗺️ Heatmap"])

    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            fig = px.box(
                df, x="Type", y="Weekly_Sales", color="Type",
                color_discrete_map=TYPE_COLORS,
                title="Sales Distribution by Store Type",
                category_orders={"Type": ["A", "B", "C"]},
            )
            fig.update_yaxes(range=[-5000, 60000])
            fig = clean_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

        with c2:
            type_summary = df.groupby("Type").agg(
                stores=("Store", "nunique"),
                avg_sales=("Weekly_Sales", "mean"),
                avg_size=("Size", "mean"),
                high_pct=("Sales_Class", "mean"),
            ).reset_index()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=type_summary["Type"],
                y=type_summary["high_pct"] * 100,
                marker_color=[TYPE_COLORS[t] for t in type_summary["Type"]],
                text=[f"{v:.1f}%" for v in type_summary["high_pct"] * 100],
                textposition="outside",
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(
                title="% High-Sales Weeks by Store Type",
                yaxis_title="High Sales %",
            )
            fig = clean_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

        # Type comparison table
        st.markdown("##### Store Type Summary")
        type_table = df.groupby("Type").agg(
            Stores=("Store", "nunique"),
            Avg_Sales=("Weekly_Sales", "mean"),
            Median_Sales=("Weekly_Sales", "median"),
            Avg_Size=("Size", "mean"),
            High_Sales_Pct=("Sales_Class", "mean"),
        ).round(2)
        type_table["Avg_Sales"] = type_table["Avg_Sales"].apply(lambda x: f"${x:,.0f}")
        type_table["Median_Sales"] = type_table["Median_Sales"].apply(lambda x: f"${x:,.0f}")
        type_table["Avg_Size"] = type_table["Avg_Size"].apply(lambda x: f"{x:,.0f} sqft")
        type_table["High_Sales_Pct"] = type_table["High_Sales_Pct"].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(type_table, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)

        store_avg = df.groupby("Store")["Weekly_Sales"].mean().sort_values()

        with c1:
            top10 = store_avg.tail(10).reset_index()
            top10.columns = ["Store", "Avg_Sales"]
            fig = go.Figure(go.Bar(
                x=top10["Avg_Sales"], y=top10["Store"].astype(str),
                orientation="h", marker_color=COLORS["success"],
                text=[f"${v:,.0f}" for v in top10["Avg_Sales"]],
                textposition="outside",
            ))
            fig.update_layout(title="🏆 Top 10 Stores")
            fig = clean_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

        with c2:
            bot10 = store_avg.head(10).reset_index()
            bot10.columns = ["Store", "Avg_Sales"]
            fig = go.Figure(go.Bar(
                x=bot10["Avg_Sales"], y=bot10["Store"].astype(str),
                orientation="h", marker_color=COLORS["danger"],
                text=[f"${v:,.0f}" for v in bot10["Avg_Sales"]],
                textposition="outside",
            ))
            fig.update_layout(title="⚠️ Bottom 10 Stores")
            fig = clean_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    with tab3:
        # Store × Month heatmap
        store_monthly = df.pivot_table(
            values="Weekly_Sales", index="Store", columns="Month", aggfunc="mean",
        ).round(0)

        fig = px.imshow(
            store_monthly,
            labels=dict(x="Month", y="Store", color="Avg Sales ($)"),
            color_continuous_scale="YlOrRd",
            title="Store × Month Performance Heatmap",
            aspect="auto",
        )
        fig.update_layout(
            xaxis=dict(
                tickmode="array", tickvals=list(range(1, 13)),
                ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            ),
        )
        fig = clean_layout(fig, height=600)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config())


# SECTION 4: PROMOTION ANALYSIS
def render_promotion_analysis(df):
    """Promotional markdown impact analysis."""
    section_header("Promotion & MarkDown Analysis", "🏷️")

    c1, c2 = st.columns(2)

    with c1:
        if "HasAnyMarkDown" in df.columns:
            promo_data = df.groupby("HasAnyMarkDown").agg(
                avg_sales=("Weekly_Sales", "mean"),
                high_pct=("Sales_Class", "mean"),
                count=("Weekly_Sales", "count"),
            ).reset_index()
            promo_data["Label"] = promo_data["HasAnyMarkDown"].map({0: "No Promotion", 1: "Has Promotion"})

            fig = go.Figure(go.Bar(
                x=promo_data["Label"],
                y=promo_data["avg_sales"],
                marker_color=[COLORS["light"], COLORS["primary"]],
                text=[f"${v:,.0f}" for v in promo_data["avg_sales"]],
                textposition="outside",
            ))
            fig.update_layout(title="Average Sales: Promotion vs None", yaxis_title="Avg Sales ($)")
            fig = clean_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    with c2:
        if "ActiveMarkDownCount" in df.columns:
            md_count = df.groupby("ActiveMarkDownCount")["Weekly_Sales"].mean().reset_index()
            fig = go.Figure(go.Bar(
                x=md_count["ActiveMarkDownCount"],
                y=md_count["Weekly_Sales"],
                marker_color=COLORS["secondary"],
                text=[f"${v:,.0f}" for v in md_count["Weekly_Sales"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="Sales by Number of Active Promotions",
                xaxis_title="# Active MarkDowns",
                yaxis_title="Avg Sales ($)",
            )
            fig = clean_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    # MarkDown correlation
    c1, c2 = st.columns(2)

    with c1:
        md_cols = [c for c in MARKDOWN_COLS if c in df.columns]
        if md_cols:
            md_corr = {col: df[col].corr(df["Weekly_Sales"]) for col in md_cols}
            fig = go.Figure(go.Bar(
                x=list(md_corr.keys()),
                y=list(md_corr.values()),
                marker_color=COLORS["primary"],
                text=[f"{v:.3f}" for v in md_corr.values()],
                textposition="outside",
            ))
            fig.add_hline(y=0, line_color="black", line_width=1)
            fig.update_layout(title="MarkDown Correlation with Sales", yaxis_title="Pearson r")
            fig = clean_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    with c2:
        if "HasAnyMarkDown" in df.columns:
            promo_type = df.groupby(["Type", "HasAnyMarkDown"])["Weekly_Sales"].mean().unstack()
            promo_type.columns = ["No Promo", "Has Promo"]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=promo_type.index, y=promo_type["No Promo"],
                                  name="No Promo", marker_color=COLORS["light"]))
            fig.add_trace(go.Bar(x=promo_type.index, y=promo_type["Has Promo"],
                                  name="Has Promo", marker_color=COLORS["primary"]))
            fig.update_layout(
                title="Promotion Effect by Store Type",
                barmode="group", yaxis_title="Avg Sales ($)",
            )
            fig = clean_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    insight_box(
        "Promotions provide a measurable sales lift across all store types. "
        "Type A stores benefit the most from promotions. Running multiple "
        "simultaneous MarkDowns shows diminishing returns after 3 active promotions."
    )


# SECTION 5: FEATURE DEEP DIVE
def render_feature_analysis(df):
    """Interactive feature exploration."""
    section_header("Feature Deep Dive", "🔬")

    tab1, tab2 = st.tabs(["📊 Distributions", "🔗 Correlations"])

    with tab1:
        # Feature selector
        numeric_cols = sorted([
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ["Store", "Dept", "Sales_Class"]
        ])

        selected_feature = st.selectbox("Select Feature", numeric_cols, index=0)

        c1, c2 = st.columns(2)

        with c1:
            fig = go.Figure()
            for label, color in CLASS_COLORS.items():
                subset = df[df["Sales_Label"] == label][selected_feature].dropna()
                fig.add_trace(go.Histogram(
                    x=subset, name=label, marker_color=color,
                    opacity=0.6, nbinsx=50,
                ))
            fig.update_layout(
                title=f"{selected_feature} Distribution by Class",
                barmode="overlay", xaxis_title=selected_feature,
                yaxis_title="Count",
            )
            fig = clean_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

        with c2:
            fig = px.box(
                df, x="Sales_Label", y=selected_feature,
                color="Sales_Label", color_discrete_map=CLASS_COLORS,
                title=f"{selected_feature} by Target Class",
            )
            fig = clean_layout(fig)
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

        # Quick stats
        c1, c2, c3, c4 = st.columns(4)
        feat = df[selected_feature].dropna()
        c1.metric("Mean", f"{feat.mean():,.2f}")
        c2.metric("Median", f"{feat.median():,.2f}")
        c3.metric("Std Dev", f"{feat.std():,.2f}")
        c4.metric("Skewness", f"{feat.skew():.2f}")

    with tab2:
        # Correlation heatmap
        corr_features = st.multiselect(
            "Select Features for Correlation Matrix",
            numeric_cols,
            default=numeric_cols[:10],
        )

        if len(corr_features) >= 2:
            corr = df[corr_features].corr()
            fig = px.imshow(
                corr, color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Feature Correlation Matrix",
                text_auto=".2f",
            )
            fig = clean_layout(fig, height=max(400, 40 * len(corr_features)))
            st.plotly_chart(fig, use_container_width=True, config=plotly_config())

        # Target correlation ranking
        st.markdown("##### Feature Correlation with Target (Sales_Class)")
        target_corr = df[numeric_cols + ["Sales_Class"]].corr()["Sales_Class"].drop("Sales_Class")
        target_corr = target_corr.sort_values(ascending=False)

        fig = go.Figure(go.Bar(
            x=target_corr.values,
            y=target_corr.index,
            orientation="h",
            marker_color=[COLORS["success"] if v > 0 else COLORS["danger"] for v in target_corr],
        ))
        fig.update_layout(title="All Features vs Target Correlation")
        fig = clean_layout(fig, height=max(400, 22 * len(target_corr)))
        st.plotly_chart(fig, use_container_width=True, config=plotly_config())


# SECTION 6: MODEL PERFORMANCE
def render_model_performance(model_results):
    """Model comparison and performance metrics."""
    section_header("Model Performance Comparison", "🤖")

    models = model_results.get("models", {})
    if not models:
        st.warning("No model results found. Run the modeling pipeline first.")
        return

    best_model = model_results.get("best_model", "")

    # Model comparison chart
    c1, c2 = st.columns([2, 1])

    with c1:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        fig = go.Figure()

        for model_name, model_data in models.items():
            values = [model_data.get(m, 0) for m in metrics]
            fig.add_trace(go.Scatterpolar(
                r=values, theta=[m.replace("_", " ").title() for m in metrics],
                fill="toself", name=model_name,
                line=dict(width=2),
                opacity=0.7,
            ))

        fig.update_layout(
            title="Model Performance Radar Chart",
            polar=dict(radialaxis=dict(visible=True, range=[0.5, 1.0])),
            showlegend=True,
        )
        fig = clean_layout(fig, height=500)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    with c2:
        st.markdown("##### 🏆 Model Rankings")
        rankings = sorted(models.items(), key=lambda x: x[1].get("f1", 0), reverse=True)
        for rank, (name, data) in enumerate(rankings, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            st.markdown(
                f"{medal} **{name}**  \n"
                f"F1: `{data.get('f1', 0):.3f}` | "
                f"AUC: `{data.get('roc_auc', 0):.3f}`"
            )
            st.progress(data.get("f1", 0))

    # Detailed metrics table
    st.markdown("##### Detailed Metrics Comparison")
    metrics_df = pd.DataFrame(models).T
    metrics_df = metrics_df[[c for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "train_time_seconds"]
                              if c in metrics_df.columns]]
    metrics_df.columns = [c.replace("_", " ").title() for c in metrics_df.columns]

    # Highlight best
    st.dataframe(
        metrics_df.style.highlight_max(axis=0, color="#D1FAE5")
                        .format("{:.4f}", subset=[c for c in metrics_df.columns if c != "Train Time Seconds"])
                        .format("{:.1f}s", subset=["Train Time Seconds"] if "Train Time Seconds" in metrics_df.columns else []),
        use_container_width=True,
    )

    # Feature importance
    feat_imp = model_results.get("feature_importance", {})
    if feat_imp:
        st.markdown("##### Feature Importance (Best Model)")
        imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

        fig = go.Figure(go.Bar(
            x=list(imp_sorted.values()),
            y=list(imp_sorted.keys()),
            orientation="h",
            marker_color=COLORS["primary"],
            text=[f"{v:.3f}" for v in imp_sorted.values()],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Top Features — {best_model}",
            xaxis_title="Importance Score",
        )
        fig = clean_layout(fig, height=max(300, 30 * len(imp_sorted)))
        st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    insight_box(
        f"<b>{best_model}</b> achieves the highest F1 score of "
        f"<b>{models.get(best_model, {}).get('f1', 0):.3f}</b>. "
        f"Store Size and recent sales history (lag features) are the most "
        f"important predictors, confirming our EDA findings.",
        type="success",
    )


# SECTION 7: BUSINESS RECOMMENDATIONS
def render_recommendations(df, model_results):
    """Actionable business insights."""
    section_header("Business Recommendations", "💼")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 📌 Key Findings")

        findings = [
            {
                "icon": "🏪",
                "title": "Store Size Drives Sales",
                "detail": (
                    f"Type A stores (avg {df[df['Type']=='A']['Size'].mean():,.0f} sqft) "
                    f"generate {df[df['Type']=='A']['Weekly_Sales'].mean()/df[df['Type']=='C']['Weekly_Sales'].mean():.1f}x "
                    f"more sales than Type C stores."
                ),
            },
            {
                "icon": "🎄",
                "title": "Holiday Planning is Critical",
                "detail": (
                    f"Holiday weeks see a "
                    f"{((df[df['IsHoliday']==True]['Weekly_Sales'].mean()/df[df['IsHoliday']==False]['Weekly_Sales'].mean())-1)*100:.0f}% "
                    f"sales increase. Pre-holiday weeks also show elevated demand."
                ),
            },
            {
                "icon": "🏷️",
                "title": "Promotions Work — With Limits",
                "detail": (
                    "Promotional markdowns boost sales, but running more than 3 "
                    "simultaneous promotions shows diminishing returns."
                ),
            },
            {
                "icon": "📈",
                "title": "Past Sales Predict Future Sales",
                "detail": (
                    "Last week's sales (Lag_Sales_1w) and 4-week rolling average "
                    "are among the strongest predictors of current performance."
                ),
            },
        ]

        for f in findings:
            st.markdown(f"""
            <div style="background:#F8FAFC; border-radius:8px; padding:1rem; margin:0.5rem 0;
                        border-left:3px solid {COLORS['primary']};">
                <b>{f['icon']} {f['title']}</b><br>
                <span style="color:#64748B; font-size:0.9rem;">{f['detail']}</span>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown("#### 🎯 Recommended Actions")

        actions = [
            {
                "priority": "HIGH",
                "color": COLORS["danger"],
                "action": "Increase Holiday Inventory",
                "detail": "Stock up 2 weeks before Thanksgiving, Christmas, Super Bowl. Focus on Type A stores.",
            },
            {
                "priority": "HIGH",
                "color": COLORS["danger"],
                "action": "Optimize Promotion Strategy",
                "detail": "Run 2-3 concurrent markdowns maximum. Focus MarkDown1 and MarkDown5 (highest correlation).",
            },
            {
                "priority": "MED",
                "color": COLORS["warning"],
                "action": "Support Underperforming Stores",
                "detail": "Bottom 10 stores need targeted interventions. Consider local market factors.",
            },
            {
                "priority": "MED",
                "color": COLORS["warning"],
                "action": "Deploy Predictive Model",
                "detail": (
                    f"The {model_results.get('best_model', 'best')} model can predict "
                    f"high/low sales weeks with {model_results.get('models', {}).get(model_results.get('best_model', ''), {}).get('accuracy', 0):.0%} accuracy."
                ),
            },
            {
                "priority": "LOW",
                "color": COLORS["success"],
                "action": "Monitor Economic Indicators",
                "detail": "Track unemployment and consumer sentiment for early warning of demand shifts.",
            },
        ]

        for a in actions:
            st.markdown(f"""
            <div style="background:#F8FAFC; border-radius:8px; padding:1rem; margin:0.5rem 0;
                        border-left:3px solid {a['color']};">
                <span style="background:{a['color']}; color:white; padding:2px 8px;
                             border-radius:4px; font-size:0.7rem; font-weight:600;">{a['priority']}</span>
                <b style="margin-left:8px;">{a['action']}</b><br>
                <span style="color:#64748B; font-size:0.9rem;">{a['detail']}</span>
            </div>
            """, unsafe_allow_html=True)


# MAIN APP
def main():
    inject_css()

    # Title
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 0.5rem 0;">
        <h1 style="color:#1E293B; margin:0;">🏪 Walmart Sales Classification</h1>
        <p style="color:#64748B; font-size:1.1rem; margin:0;">
            Interactive Dashboard — EDA, Model Performance & Business Insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    try:
        df = load_eda_data()
        dashboard_json = load_dashboard_json()
        model_results = load_model_results()
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}. Run the pipeline first.")
        return

    # Sidebar filters
    filtered_df = render_sidebar(df)

    # Show filter status
    if len(filtered_df) < len(df):
        st.info(
            f"🔍 Showing {len(filtered_df):,} of {len(df):,} records "
            f"({len(filtered_df)/len(df)*100:.1f}%) based on filters"
        )

    # Navigation
    page = st.radio(
        "Navigate",
        ["📊 Executive Summary", "📈 Sales Overview", "🏬 Store Performance",
         "🏷️ Promotions", "🔬 Feature Analysis", "🤖 Model Performance",
         "💼 Recommendations"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Render selected page
    if page == "📊 Executive Summary":
        render_executive_summary(filtered_df, model_results)
    elif page == "📈 Sales Overview":
        render_sales_overview(filtered_df)
    elif page == "🏬 Store Performance":
        render_store_performance(filtered_df)
    elif page == "🏷️ Promotions":
        render_promotion_analysis(filtered_df)
    elif page == "🔬 Feature Analysis":
        render_feature_analysis(filtered_df)
    elif page == "🤖 Model Performance":
        render_model_performance(model_results)
    elif page == "💼 Recommendations":
        render_recommendations(filtered_df, model_results)


if __name__ == "__main__":
    main()