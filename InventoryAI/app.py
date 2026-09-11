import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="InventoryAI",
    page_icon="📦",
    layout="wide"
)

# =========================================================
# PREMIUM UI STYLING
# =========================================================

st.markdown("""
<style>
    /* ---------- Global ---------- */
    .stApp {
        background: #f7f8fc;
    }

    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
        scrollbar-color: #475569 #111827;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] .stFileUploader {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 14px;
        padding: 8px;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 10px;
    }

    [data-testid="stSidebar"] hr {
        border-color: #374151;
    }


    /* ---------- Scrollbars ---------- */
    ::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }

    ::-webkit-scrollbar-track {
        background: #f7f8fc;
    }

    ::-webkit-scrollbar-thumb {
        background: #94a3b8;
        border-radius: 999px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }

    /* Streamlit sidebar scroll area */
    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"]::-webkit-scrollbar {
        width: 7px;
    }

    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"]::-webkit-scrollbar-track {
        background: #111827;
    }

    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {
        background: #64748b;
        border-radius: 999px;
        border: 1px solid #111827;
    }

    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"]::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }

    /* Firefox */
    section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
        scrollbar-color: #64748b #111827;
        scrollbar-width: thin;
    }


    /* ---------- Sidebar parameter layout ---------- */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] ul {
        margin-top: 0.15rem;
        padding-left: 1.05rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
        margin-bottom: 0.08rem;
        font-size: 12px;
        line-height: 1.45;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.35rem;
    }

    /* ---------- Hero ---------- */
    .hero {
        background: linear-gradient(135deg, #111827 0%, #1e3a5f 55%, #2563eb 100%);
        border-radius: 22px;
        padding: 24px 32px;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 12px 35px rgba(15, 23, 42, 0.12);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.20);
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: .2px;
        margin-bottom: 14px;
    }

    .hero h1 {
        margin: 0;
        font-size: 38px;
        line-height: 1.1;
        font-weight: 750;
        letter-spacing: -1px;
    }

    .hero p {
        margin: 10px 0 0;
        color: #dbeafe;
        font-size: 16px;
    }

    .hero-author {
        margin-top: 20px;
        font-size: 16px;
        font-weight: 800;
        color: #cbd5e1;
        letter-spacing: .1px;
    }

    .hero-author strong,
    .hero-author a {
        color: #cbd5e1 !important;
        text-decoration: none;
        font-weight: 800;
    }

    .hero-author a:hover {
        color: #e2e8f0 !important;
        text-decoration: underline;
    }

    /* ---------- Section titles ---------- */
    .section-title {
        font-size: 23px;
        font-weight: 720;
        color: #111827;
        margin: 10px 0 4px;
        letter-spacing: -0.3px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 16px;
    }

    /* ---------- KPI cards ---------- */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin: 8px 0 28px;
    }

    .kpi-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 17px;
        padding: 19px 20px;
        min-height: 118px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.045);
        position: relative;
        overflow: hidden;
    }

    .kpi-card:after {
        content: "";
        position: absolute;
        left: 0;
        bottom: 0;
        height: 3px;
        width: 100%;
        background: linear-gradient(90deg, #2563eb, #60a5fa);
    }

    .kpi-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 650;
        margin-bottom: 9px;
    }

    .kpi-value {
        color: #0f172a;
        font-size: 29px;
        font-weight: 760;
        letter-spacing: -0.7px;
    }

    .kpi-icon {
        float: right;
        font-size: 20px;
        opacity: .9;
    }

    /* ---------- Panels ---------- */
    .panel {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px 20px 12px;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.035);
        margin-bottom: 18px;
    }

    /* ---------- Buttons ---------- */
    .stDownloadButton button {
        border-radius: 10px;
        font-weight: 650;
        border: 1px solid #2563eb;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding: 18px 0 5px;
    }

    @media (max-width: 900px) {
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .hero h1 {
            font-size: 30px;
        }
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DEMO DATA
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "InventoryAI/data/InventoryAI_Final_Dataset.csv"
    )


# =========================================================
# LOAD XGBOOST MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = XGBRegressor()

    model.load_model(
        "InventoryAI/model/inventoryai_xgb_model.json"
    )

    return model


# =========================================================
# MODEL FEATURES
# =========================================================

MODEL_FEATURES = [
    "Year",
    "Month",
    "Day",
    "Day_of_Week",
    "Inventory Level",
    "Price",
    "Discount",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Lag_28",
    "Rolling_7",
    "Rolling_28"
]


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_features(df):

    df = df.copy()

    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Date"]
    )

    # -----------------------------------------------------
    # OPTIONAL COLUMNS
    # -----------------------------------------------------

    if "Category" not in df.columns:

        df["Category"] = "Unknown"

    if "Price" not in df.columns:

        df["Price"] = 0

    if "Discount" not in df.columns:

        df["Discount"] = 0

    # -----------------------------------------------------
    # SORT DATA
    # -----------------------------------------------------

    df = df.sort_values(
        [
            "Store ID",
            "Product ID",
            "Date"
        ]
    ).copy()

    # -----------------------------------------------------
    # TIME FEATURES
    # -----------------------------------------------------

    df["Year"] = df["Date"].dt.year

    df["Month"] = df["Date"].dt.month

    df["Day"] = df["Date"].dt.day

    df["Day_of_Week"] = (
        df["Date"].dt.dayofweek
    )

    # -----------------------------------------------------
    # LAG FEATURES
    # -----------------------------------------------------

    grouped_sales = df.groupby(
        [
            "Store ID",
            "Product ID"
        ]
    )["Units Sold"]

    df["Lag_1"] = (
        grouped_sales.shift(1)
    )

    df["Lag_7"] = (
        grouped_sales.shift(7)
    )

    df["Lag_14"] = (
        grouped_sales.shift(14)
    )

    df["Lag_28"] = (
        grouped_sales.shift(28)
    )

    # -----------------------------------------------------
    # ROLLING FEATURES
    # -----------------------------------------------------

    df["Rolling_7"] = (

        df.groupby(
            [
                "Store ID",
                "Product ID"
            ]
        )["Units Sold"]

        .transform(
            lambda x:
            x.shift(1)
            .rolling(7)
            .mean()
        )
    )

    df["Rolling_28"] = (

        df.groupby(
            [
                "Store ID",
                "Product ID"
            ]
        )["Units Sold"]

        .transform(
            lambda x:
            x.shift(1)
            .rolling(28)
            .mean()
        )
    )

    return df


# =========================================================
# INVENTORY DECISION ENGINE
# =========================================================

def create_inventory_intelligence(
    df,
    model
):

    df = df.copy()

    # -----------------------------------------------------
    # XGBOOST PREDICTION
    # -----------------------------------------------------

    model_input = df[
        MODEL_FEATURES
    ]

    df["Predicted Units Sold"] = (
        model.predict(model_input)
    )

    # Prevent negative predictions

    df["Predicted Units Sold"] = (
        df["Predicted Units Sold"]
        .clip(lower=0)
    )

    # -----------------------------------------------------
    # FORECAST WEEKLY DEMAND
    # -----------------------------------------------------

    df["Forecasted Weekly Demand"] = (

        df["Predicted Units Sold"] * 7

    )

    # -----------------------------------------------------
    # INVENTORY COVERAGE %
    #
    # Current Inventory
    # ------------------------- × 100
    # Forecasted Weekly Demand
    # -----------------------------------------------------

    df["Inventory_Coverage_Percent"] = (

        np.where(

            df["Forecasted Weekly Demand"] > 0,

            (
                df["Inventory Level"]
                /
                df["Forecasted Weekly Demand"]
            ) * 100,

            100

        )

    )

    # -----------------------------------------------------
    # RISK CLASSIFICATION
    #
    # < 25%       → HIGH
    # 25% - <75%  → MEDIUM
    # >= 75%      → LOW
    # -----------------------------------------------------

    def classify_risk(
        coverage
    ):

        if coverage < 25:

            return "High"

        elif coverage < 75:

            return "Medium"

        else:

            return "Low"

    df["Risk_Level"] = (

        df[
            "Inventory_Coverage_Percent"
        ]

        .apply(
            classify_risk
        )

    )

    # -----------------------------------------------------
    # RECOMMENDED ORDER
    #
    # Weekly Forecast - Current Inventory
    # -----------------------------------------------------

    df["Recommended_Order_Qty"] = (

        df["Forecasted Weekly Demand"]
        -
        df["Inventory Level"]

    ).clip(
        lower=0
    )

    df["Recommended_Order_Qty"] = (

        np.ceil(
            df["Recommended_Order_Qty"]
        )

        .astype(int)

    )

    # -----------------------------------------------------
    # BUSINESS RECOMMENDATION
    # -----------------------------------------------------

    def recommendation(
        row
    ):

        if row["Risk_Level"] == "High":

            return "Urgent Reorder"

        elif row["Risk_Level"] == "Medium":

            return "Plan Reorder"

        else:

            return "Healthy Stock"

    df["Recommendation"] = (

        df.apply(
            recommendation,
            axis=1
        )

    )

    return df


# =========================================================
# LOAD MODEL
# =========================================================

model = load_model()


# =========================================================
# PREMIUM HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-POWERED RETAIL INTELLIGENCE</div>
    <h1>InventoryAI</h1>
    <p>Demand forecasting, inventory risk detection and smarter replenishment decisions — all in one workspace.</p>
    <div class="hero-author">
        Built by <strong>Pradeep Kalasagond</strong>
        &nbsp;·&nbsp;
        <a href="https://www.linkedin.com/in/pradeep-kalasagond-95579a230/" target="_blank">LinkedIn ↗</a>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR — DATA & FILTERS
# =========================================================

st.sidebar.markdown("""
<div style="padding: 6px 2px 20px;">
    <div style="font-size:24px;font-weight:800;">📦 InventoryAI</div>
    <div style="font-size:12px;color:#94a3b8;margin-top:4px;">
        Retail Inventory Intelligence
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 📂 Data Workspace")

st.sidebar.markdown(
    '<div style="font-size:13px;color:#cbd5e1;margin-bottom:8px;">Upload Retail File</div>',
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader(
    "Choose CSV or Excel",
    type=["csv", "xlsx", "xls"],
    label_visibility="collapsed",
    help="Upload a CSV or Excel file using the InventoryAI data schema."
)

# Upload schema — intentionally shown directly below the uploader.
st.sidebar.markdown("### 📋 Upload Parameters")

param_left, param_right = st.sidebar.columns(2, gap="small")

with param_left:
    st.markdown("**Required Columns**")
    st.markdown("""
    - Date
    - Store ID
    - Product ID
    - Inventory Level
    - Units Sold
    """)

with param_right:
    st.markdown("**Recommended Columns**")
    st.markdown("""
    - Category
    - Region
    - Price
    - Discount
    """)

st.sidebar.markdown(
    '<div style="font-size:11px;color:#64748b;margin-top:9px;">'
    'CSV · XLSX · XLS &nbsp;•&nbsp; Daily historical data recommended'
    '</div>',
    unsafe_allow_html=True
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.sidebar.success("Dataset uploaded successfully")
        is_uploaded_data = True

    except Exception as e:
        st.error(f"Unable to read the uploaded file: {e}")
        st.stop()

else:
    df = load_data()
    is_uploaded_data = False

st.sidebar.divider()
st.sidebar.markdown("### 🔎 Filters")

# =========================================================
# BASIC DATA CHECK
# =========================================================

core_columns = [
    "Date",
    "Store ID",
    "Product ID",
    "Inventory Level",
    "Units Sold"
]

missing_core = [
    column for column in core_columns
    if column not in df.columns
]

if missing_core:
    st.error(
        "This dataset is missing required columns: "
        + ", ".join(missing_core)
    )
    st.info(
        "Required schema: Date, Store ID, Product ID, "
        "Inventory Level and Units Sold."
    )
    st.stop()


# =========================================================
# PROCESS UPLOADED RAW DATA
# =========================================================

if is_uploaded_data:

    with st.spinner(
        "⚙️ Preparing dataset and generating features..."
    ):

        # Feature engineering

        df = create_features(
            df
        )

        # Remove rows without enough history

        df = df.dropna(
            subset=[
                "Lag_1",
                "Lag_7",
                "Lag_14",
                "Lag_28",
                "Rolling_7",
                "Rolling_28"
            ]
        ).copy()

        if len(df) == 0:

            st.error(
                "❌ Not enough historical data to "
                "generate the required demand features."
            )

            st.stop()

        # Generate predictions + decisions

        df = create_inventory_intelligence(
            df,
            model
        )


# =========================================================
# PROCESS DEMO DATASET
# =========================================================

else:

    # -----------------------------------------------------
    # DEMO DATA ALREADY CONTAINS PREDICTIONS
    # -----------------------------------------------------

    if "Predicted Units Sold" not in df.columns:

        st.error(
            "❌ Demo dataset does not contain "
            "Predicted Units Sold."
        )

        st.stop()

    # -----------------------------------------------------
    # WEEKLY FORECAST
    # -----------------------------------------------------

    df["Forecasted Weekly Demand"] = (

        df["Predicted Units Sold"] * 7

    )

    # -----------------------------------------------------
    # INVENTORY COVERAGE %
    # -----------------------------------------------------

    df["Inventory_Coverage_Percent"] = (

        np.where(

            df["Forecasted Weekly Demand"] > 0,

            (
                df["Inventory Level"]
                /
                df["Forecasted Weekly Demand"]
            ) * 100,

            100

        )

    )

    # -----------------------------------------------------
    # RECALCULATE RISK
    # -----------------------------------------------------

    def classify_demo_risk(
        coverage
    ):

        if coverage < 25:

            return "High"

        elif coverage < 75:

            return "Medium"

        else:

            return "Low"

    df["Risk_Level"] = (

        df[
            "Inventory_Coverage_Percent"
        ]

        .apply(
            classify_demo_risk
        )

    )

    # -----------------------------------------------------
    # RECALCULATE ORDER QUANTITY
    # -----------------------------------------------------

    df["Recommended_Order_Qty"] = (

        df["Forecasted Weekly Demand"]
        -
        df["Inventory Level"]

    ).clip(
        lower=0
    )

    df["Recommended_Order_Qty"] = (

        np.ceil(
            df["Recommended_Order_Qty"]
        )

        .astype(int)

    )

    # -----------------------------------------------------
    # RECALCULATE RECOMMENDATION
    # -----------------------------------------------------

    def demo_recommendation(
        row
    ):

        if row["Risk_Level"] == "High":

            return "Urgent Reorder"

        elif row["Risk_Level"] == "Medium":

            return "Plan Reorder"

        else:

            return "Healthy Stock"

    df["Recommendation"] = (

        df.apply(
            demo_recommendation,
            axis=1
        )

    )


# =========================================================
# CURRENT INVENTORY POSITION
# =========================================================
#
# Keep ONLY the latest record for each
# Store + Product combination.
#
# Example:
#
# S001 + P001 → latest available date
# S001 + P002 → latest available date
# ...
#
# This gives us the current inventory position.
# =========================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df = (

    df

    .sort_values(
        [
            "Store ID",
            "Product ID",
            "Date"
        ]
    )

    .groupby(
        [
            "Store ID",
            "Product ID"
        ],
        as_index=False
    )

    .tail(1)

    .reset_index(
        drop=True
    )

)


# =========================================================
# STORE FILTER
# =========================================================

store_options = sorted(
    df["Store ID"]
    .dropna()
    .unique()
)

selected_store = (
    st.sidebar.selectbox(
        "Store",
        ["All"] + list(
            store_options
        )
    )
)


# =========================================================
# CATEGORY FILTER
# =========================================================

category_options = sorted(
    df["Category"]
    .dropna()
    .unique()
)

selected_category = (
    st.sidebar.selectbox(
        "Category",
        ["All"] + list(
            category_options
        )
    )
)


# =========================================================
# RISK FILTER
# =========================================================

risk_options = [
    "All",
    "High",
    "Medium",
    "Low"
]

selected_risk = (
    st.sidebar.selectbox(
        "Risk Level",
        risk_options
    )
)


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_df = df.copy()


if selected_store != "All":

    filtered_df = (
        filtered_df[
            filtered_df["Store ID"]
            == selected_store
        ]
    )


if selected_category != "All":

    filtered_df = (
        filtered_df[
            filtered_df["Category"]
            == selected_category
        ]
    )


if selected_risk != "All":

    filtered_df = (
        filtered_df[
            filtered_df["Risk_Level"]
            == selected_risk
        ]
    )


# =========================================================
# KPI CALCULATIONS
# =========================================================

total_inventory = (

    filtered_df[
        "Inventory Level"
    ]

    .sum()

)


forecasted_weekly_demand = (

    filtered_df[
        "Forecasted Weekly Demand"
    ]

    .sum()

)


high_risk_items = (

    filtered_df[
        "Risk_Level"
    ]

    .eq("High")

    .sum()

)


recommended_order = (

    filtered_df[
        "Recommended_Order_Qty"
    ]

    .sum()

)


avg_coverage = (

    filtered_df[
        "Inventory_Coverage_Percent"
    ]

    .mean()

)


store_count = (

    filtered_df[
        "Store ID"
    ]

    .nunique()

)


# =========================================================
# EXECUTIVE KPI STRIP
# =========================================================

def fmt_number(value):
    return f"{value:,.0f}"

st.markdown("""
<div class="section-title">Executive Overview</div>
<div class="section-subtitle">
    A real-time snapshot of inventory health and replenishment priorities.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-icon">📦</div>
        <div class="kpi-label">CURRENT INVENTORY</div>
        <div class="kpi-value">{fmt_number(total_inventory)}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">📈</div>
        <div class="kpi-label">WEEKLY DEMAND</div>
        <div class="kpi-value">{fmt_number(forecasted_weekly_demand)}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">🚨</div>
        <div class="kpi-label">HIGH-RISK PRODUCTS</div>
        <div class="kpi-value">{high_risk_items:,}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">🛒</div>
        <div class="kpi-label">RECOMMENDED ORDER</div>
        <div class="kpi-value">{fmt_number(recommended_order)}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">◒</div>
        <div class="kpi-label">AVG. INVENTORY COVERAGE</div>
        <div class="kpi-value">{avg_coverage:.1f}%</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">🏪</div>
        <div class="kpi-label">ACTIVE STORES</div>
        <div class="kpi-value">{store_count:,}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# =========================================================
# INVENTORY OVERVIEW + RISK
# =========================================================

st.markdown("""
<div class="section-title">Inventory Health</div>
<div class="section-subtitle">
    Current Store × Product positions after applying your selected filters.
</div>
""", unsafe_allow_html=True)

st.markdown(
    f'<div class="panel"><strong>{len(filtered_df):,}</strong> current Store-Product positions are being analyzed.</div>',
    unsafe_allow_html=True
)

risk_summary = (
    filtered_df["Risk_Level"]
    .value_counts()
    .reindex(["High", "Medium", "Low"], fill_value=0)
    .rename_axis("Risk Level")
    .reset_index(name="Items")
)

c1, c2 = st.columns([1.25, 1], gap="large")

with c1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 🚨 Inventory Risk Distribution")
    fig = px.bar(
        risk_summary,
        x="Risk Level",
        y="Items",
        text="Items",
        color="Risk Level",
        color_discrete_map={
            "High": "#ef4444",
            "Medium": "#f59e0b",
            "Low": "#10b981"
        }
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title=None,
        yaxis_title="Products",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    high_count = int(risk_summary.loc[risk_summary["Risk Level"] == "High", "Items"].iloc[0])
    medium_count = int(risk_summary.loc[risk_summary["Risk Level"] == "Medium", "Items"].iloc[0])
    low_count = int(risk_summary.loc[risk_summary["Risk Level"] == "Low", "Items"].iloc[0])

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Action Summary")
    st.metric("Urgent Reorder", f"{high_count:,}")
    st.metric("Plan Reorder", f"{medium_count:,}")
    st.metric("Healthy Stock", f"{low_count:,}")
    st.caption("Risk is calculated from inventory coverage against forecasted weekly demand.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# ADDITIONAL INVENTORY VISUALIZATIONS
# =========================================================

v1, v2 = st.columns(2, gap="large")

with v1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 🏷️ Category Risk")
    st.caption("Categories with the highest number of inventory-risk positions.")

    category_risk = (
        filtered_df[filtered_df["Risk_Level"].isin(["High", "Medium"])]
        .groupby("Category")
        .size()
        .reset_index(name="Risk Items")
        .sort_values("Risk Items", ascending=False)
        .head(10)
    )

    if len(category_risk) > 0:
        fig_category_risk = px.bar(
            category_risk,
            x="Risk Items",
            y="Category",
            orientation="h",
            text="Risk Items"
        )
        fig_category_risk.update_traces(textposition="outside")
        fig_category_risk.update_layout(
            template="plotly_white",
            height=360,
            margin=dict(l=20, r=35, t=20, b=20),
            xaxis_title="Risk Items",
            yaxis_title=None,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(
            fig_category_risk,
            use_container_width=True,
            config={"displayModeBar": False}
        )
    else:
        st.info("No High or Medium risk items under the selected filters.")

    st.markdown('</div>', unsafe_allow_html=True)

with v2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("#### 📦 Inventory vs Forecasted Demand")
    st.caption("Current inventory compared with expected weekly demand.")

    comparison = filtered_df[
        ["Product ID", "Inventory Level", "Forecasted Weekly Demand"]
    ].copy()

    comparison["Product"] = comparison["Product ID"].astype(str)
    comparison = comparison.sort_values(
        "Forecasted Weekly Demand",
        ascending=False
    ).head(10)

    comparison_long = comparison.melt(
        id_vars=["Product"],
        value_vars=["Inventory Level", "Forecasted Weekly Demand"],
        var_name="Metric",
        value_name="Units"
    )

    comparison_long["Metric"] = comparison_long["Metric"].replace({
        "Inventory Level": "Current Inventory",
        "Forecasted Weekly Demand": "Weekly Demand"
    })

    fig_comparison = px.bar(
        comparison_long,
        x="Product",
        y="Units",
        color="Metric",
        barmode="group"
    )
    fig_comparison.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title=None,
        yaxis_title="Units",
        legend_title=None,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(
        fig_comparison,
        use_container_width=True,
        config={"displayModeBar": False}
    )

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# TOP 100 REORDER ITEMS
# =========================================================

st.markdown("""
<div class="section-title">🛒 Priority Actions</div>
<div class="section-subtitle">
    The 100 products with the largest recommended replenishment quantities.
</div>
""", unsafe_allow_html=True)

top_actions = (

    filtered_df[

        [
            "Date",
            "Store ID",
            "Product ID",
            "Category",
            "Inventory Level",
            "Forecasted Weekly Demand",
            "Inventory_Coverage_Percent",
            "Risk_Level",
            "Recommended_Order_Qty",
            "Recommendation"
        ]

    ]

    .sort_values(
        "Recommended_Order_Qty",
        ascending=False
    )

    .head(100)

)


st.dataframe(
    top_actions,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Inventory Level": st.column_config.NumberColumn(format="%,.0f"),
        "Forecasted Weekly Demand": st.column_config.NumberColumn(format="%,.0f"),
        "Inventory_Coverage_Percent": st.column_config.NumberColumn(format="%.1f%%"),
        "Recommended_Order_Qty": st.column_config.NumberColumn(format="%,.0f"),
    }
)


# =========================================================
# COMPLETE INVENTORY DECISION LIST
# =========================================================

st.markdown("""
<div class="section-title">📋 Complete Inventory Decision List</div>
<div class="section-subtitle">
    Full current inventory position with forecast, risk and recommended action.
</div>
""", unsafe_allow_html=True)

complete_inventory = (

    filtered_df[

        [
            "Date",
            "Store ID",
            "Product ID",
            "Category",
            "Inventory Level",
            "Forecasted Weekly Demand",
            "Inventory_Coverage_Percent",
            "Risk_Level",
            "Recommended_Order_Qty",
            "Recommendation"
        ]

    ]

    .sort_values(
        "Recommended_Order_Qty",
        ascending=False
    )

)


st.write(
    f"Showing **{len(complete_inventory):,} "
    "Store-Product positions."
)


st.dataframe(
    complete_inventory,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Inventory Level": st.column_config.NumberColumn(format="%,.0f"),
        "Forecasted Weekly Demand": st.column_config.NumberColumn(format="%,.0f"),
        "Inventory_Coverage_Percent": st.column_config.NumberColumn(format="%.1f%%"),
        "Recommended_Order_Qty": st.column_config.NumberColumn(format="%,.0f"),
    }
)


# =========================================================
# DOWNLOAD COMPLETE REPORT
# =========================================================

st.markdown("""
<div class="section-title">⬇️ Export</div>
<div class="section-subtitle">
    Download the complete filtered inventory decision report.
</div>
""", unsafe_allow_html=True)

download_data = (

    complete_inventory

    .to_csv(
        index=False
    )

    .encode(
        "utf-8"
    )

)


st.download_button(
    label="⬇️ Download Complete Inventory Report",
    data=download_data,
    file_name="InventoryAI_Complete_Report.csv",
    mime="text/csv"
)


# =========================================================
# CATEGORY SUMMARY
# =========================================================

st.markdown("""
<div class="section-title">🏷️ Category Intelligence</div>
<div class="section-subtitle">
    Compare inventory levels, forecast demand and replenishment needs across categories.
</div>
""", unsafe_allow_html=True)

category_summary = (

    filtered_df

    .groupby(
        "Category"
    )

    .agg(

        Total_Inventory=(
            "Inventory Level",
            "sum"
        ),

        Avg_Weekly_Forecast=(
            "Forecasted Weekly Demand",
            "mean"
        ),

        Avg_Inventory_Coverage=(
            "Inventory_Coverage_Percent",
            "mean"
        ),

        Recommended_Order_Qty=(
            "Recommended_Order_Qty",
            "sum"
        )

    )

    .reset_index()

)


st.dataframe(
    category_summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Total_Inventory": st.column_config.NumberColumn(format="%,.0f"),
        "Avg_Weekly_Forecast": st.column_config.NumberColumn(format="%,.0f"),
        "Avg_Inventory_Coverage": st.column_config.NumberColumn(format="%.1f%%"),
        "Recommended_Order_Qty": st.column_config.NumberColumn(format="%,.0f"),
    }
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "InventoryAI | XGBoost-powered Retail Inventory "
    "Intelligence Platform"
)
