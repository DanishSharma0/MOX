"""
app.py — Streamlit Dashboard
E-commerce Sales Prediction & Analytics System

Sections:
  A. Dashboard Overview
  B. Dataset Management
  C. Model Training & Evaluation
  D. Sales Forecasting
  E. Product & Sales Analytics
  F. Prediction History
  G. Project Information
"""
import json
import logging
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config
from src import database as db
from src.data_loader import load_raw_data, dataset_summary
from src.cleaning import clean_transactions, aggregate_daily_revenue, cleaning_summary
from src.feature_engineering import (
    build_features,
    chronological_split,
    get_feature_columns,
    add_classification_target,
)
from src.train_regression import run_all_regression
from src.train_classification import run_all_classification
from src.evaluation import (
    save_regression_metrics_csv,
    save_classification_metrics_csv,
    plot_model_comparison,
    plot_sales_trend,
    save_evaluation_summary,
)
from src.forecasting import generate_forecast, save_forecast_csv
from src.analytics import top_products_from_df, monthly_revenue_from_df

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MOX · E-commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — premium dark theme with glassmorphism & micro-animations
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* Global reset */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

/* App background */
.stApp {
    background: linear-gradient(135deg, #0a0a14 0%, #0d0f1e 40%, #0a1628 100%);
    background-attachment: fixed;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0f22 0%, #10142b 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.9rem !important;
    padding: 0.5rem 0.75rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    cursor: pointer;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,102,241,0.15) !important;
    color: #a5b4fc !important;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #1e3a5f 50%, #0c1a2e 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #e879f9, #fb7185);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin-right: 0.4rem;
    margin-top: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    backdrop-filter: blur(12px);
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.2);
    border-color: rgba(99,102,241,0.45);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.card-indigo::before  { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.kpi-card.card-pink::before    { background: linear-gradient(90deg, #ec4899, #f43f5e); }
.kpi-card.card-teal::before    { background: linear-gradient(90deg, #14b8a6, #06b6d4); }
.kpi-card.card-amber::before   { background: linear-gradient(90deg, #f59e0b, #f97316); }
.kpi-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.1;
    margin-bottom: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}

/* Section headers */
.section-title {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1.8rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99,102,241,0.2);
}
.section-title .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #ec4899);
    flex-shrink: 0;
}

/* Page pill */
.page-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    color: #a5b4fc;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    margin-bottom: 1.2rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Status badges */
.status-online {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.35);
    color: #34d399;
    font-size: 0.78rem; font-weight: 600;
    padding: 0.3rem 0.7rem; border-radius: 999px;
}
.status-offline {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.35);
    color: #f87171;
    font-size: 0.78rem; font-weight: 600;
    padding: 0.3rem 0.7rem; border-radius: 999px;
}

/* Sidebar brand */
.sidebar-brand {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.sidebar-tagline {
    font-size: 0.72rem;
    color: #475569;
    margin-bottom: 1.2rem;
}

/* Info/warning/success boxes */
.custom-info {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #c7d2fe;
}
.custom-success {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #6ee7b7;
}
.custom-warning {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #fcd34d;
}

/* Streamlit overrides */
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}
div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f1e; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* Project info table */
.info-table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.88rem;
}
.info-table th {
    background: rgba(99,102,241,0.15);
    color: #a5b4fc;
    padding: 0.6rem 1rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.info-table td {
    padding: 0.6rem 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    color: #cbd5e1;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}
.info-table tr:last-child td { border-bottom: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme helper
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,15,34,0.6)",
    font=dict(family="Inter", color="#94a3b8", size=12),
    xaxis=dict(
        gridcolor="rgba(99,102,241,0.08)",
        zerolinecolor="rgba(99,102,241,0.12)",
        linecolor="rgba(99,102,241,0.2)",
    ),
    yaxis=dict(
        gridcolor="rgba(99,102,241,0.08)",
        zerolinecolor="rgba(99,102,241,0.12)",
        linecolor="rgba(99,102,241,0.2)",
    ),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(99,102,241,0.2)",
        borderwidth=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(
        bgcolor="#1e1b4b",
        bordercolor="rgba(99,102,241,0.5)",
        font=dict(family="Inter", size=12),
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────

def ss_get(key, default=None):
    return st.session_state.get(key, default)


def _load_daily_df():
    """Load daily df from session or MongoDB."""
    if ss_get("daily_df") is not None:
        return ss_get("daily_df")
    recs = db.get_all_sales()
    if recs:
        df = pd.DataFrame(recs)
        df["date"] = pd.to_datetime(df["date"])
        st.session_state["daily_df"] = df
        return df
    return None


def _download_dataset() -> bool:
    """
    Download the UCI Online Retail dataset ZIP and extract the xlsx.
    Returns True on success, False on failure.
    """
    import io, zipfile, requests
    UCI_ZIP = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
    os.makedirs(config.DATA_DIR, exist_ok=True)
    try:
        progress_bar = st.progress(0, text="Connecting to UCI repository …")
        resp = requests.get(UCI_ZIP, stream=True, timeout=60)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        chunks = []
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=65536):
            chunks.append(chunk)
            downloaded += len(chunk)
            if total:
                pct = min(int(downloaded / total * 90), 90)
                progress_bar.progress(pct, text=f"Downloading … {downloaded // 1024:,} KB")

        progress_bar.progress(92, text="Extracting …")
        z = zipfile.ZipFile(io.BytesIO(b"".join(chunks)))
        for name in z.namelist():
            if name.lower().endswith(".xlsx"):
                data = z.read(name)
                with open(config.DATASET_PATH, "wb") as f:
                    f.write(data)
                progress_bar.progress(100, text="✅ Download complete!")
                return True

        progress_bar.empty()
        st.error("ZIP downloaded but no .xlsx file found inside.")
        return False
    except Exception as exc:
        st.error(f"Download failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🛒 E-commerce Sales Analytics</div>
  <div class="hero-subtitle">Intelligent sales prediction, forecasting &amp; business intelligence</div>
  <div style="margin-top:0.8rem;">
    <span class="hero-badge">MongoDB</span>
    <span class="hero-badge">Scikit-learn</span>
    <span class="hero-badge">Streamlit</span>
    <span class="hero-badge">PyMongo</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-brand">MOX Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">E-commerce Intelligence Platform</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard Overview",
            "📂 Dataset Management",
            "🤖 Model Training",
            "🔮 Sales Forecasting",
            "📈 Analytics",
            "📜 Prediction History",
            "ℹ️ Project Info",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    hc = db.health_check()
    if hc["ok"]:
        st.markdown('<div class="status-online">🟢 MongoDB Connected</div>', unsafe_allow_html=True)
        st.caption(f"DB · `{config.DB_NAME}`")
    else:
        st.markdown('<div class="status-offline">🔴 DB Offline</div>', unsafe_allow_html=True)
        st.caption(hc["message"])

    st.markdown("---")
    st.caption(f"🕐 {datetime.now().strftime('%b %d, %Y · %H:%M')}")


def section_title(label: str):
    st.markdown(
        f'<div class="section-title"><span class="dot"></span>{label}</div>',
        unsafe_allow_html=True,
    )


def page_pill(label: str):
    st.markdown(f'<div class="page-pill">{label}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# A. DASHBOARD OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

if page == "📊 Dashboard Overview":
    page_pill("📊 Dashboard Overview")

    hc = db.health_check()
    daily_df = _load_daily_df()

    if daily_df is None or daily_df.empty:
        st.markdown(
            '<div class="custom-info">ℹ️ No sales data found. Please go to <strong>📂 Dataset Management</strong> and import the dataset first.</div>',
            unsafe_allow_html=True,
        )
    else:
        stats = db.get_total_stats()
        total_rev = stats.get("total_revenue", daily_df["revenue"].sum())
        total_qty = stats.get("total_quantity", daily_df["quantity"].sum())
        total_tx  = stats.get("total_transactions", daily_df["num_transactions"].sum())
        record_ct = stats.get("record_count", len(daily_df))

        c1, c2, c3, c4 = st.columns(4)
        cards = [
            (c1, "card-indigo", "💰", f"£{total_rev:,.0f}", "Total Revenue"),
            (c2, "card-pink",   "📦", f"{total_qty:,.0f}", "Units Sold"),
            (c3, "card-teal",   "🧾", f"{total_tx:,.0f}", "Transactions"),
            (c4, "card-amber",  "📅", f"{record_ct:,}", "Days of Data"),
        ]
        for col, cls, icon, val, lbl in cards:
            col.markdown(
                f'<div class="kpi-card {cls}">'
                f'<div class="kpi-icon">{icon}</div>'
                f'<div class="kpi-value">{val}</div>'
                f'<div class="kpi-label">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        section_title("Daily Revenue Trend")
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_df["date"],
                y=daily_df["revenue"],
                mode="lines",
                line=dict(color="#818cf8", width=2),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.08)",
                name="Revenue",
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Revenue: £%{y:,.0f}<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                              xaxis_title="Date", yaxis_title="Revenue (£)")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(daily_df["date"], daily_df["revenue"], color="#818cf8", linewidth=1.5)
            ax.fill_between(daily_df["date"], daily_df["revenue"], alpha=0.12, color="#818cf8")
            ax.set_facecolor("#0d0f22"); fig.patch.set_facecolor("#0d0f22")
            ax.tick_params(colors="#94a3b8"); ax.spines[:].set_color("#1e293b")
            ax.grid(axis="y", linestyle="--", alpha=0.2, color="#334155")
            st.pyplot(fig); plt.close()

        section_title("Latest Model Metrics")
        saved_metrics = db.get_model_metrics()
        if saved_metrics:
            st.dataframe(pd.DataFrame(saved_metrics), use_container_width=True)
        else:
            st.markdown(
                '<div class="custom-info">ℹ️ No model metrics saved yet. Train models in <strong>🤖 Model Training</strong>.</div>',
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# B. DATASET MANAGEMENT
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📂 Dataset Management":
    page_pill("📂 Dataset Management")

    dataset_exists = os.path.exists(config.DATASET_PATH)
    if dataset_exists:
        st.markdown(
            '<div class="custom-success">✅ Dataset file found and ready to import.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="custom-warning">⚠️ Dataset file not found on this server. '
            'Click <strong>Auto-Download</strong> below to fetch it automatically from UCI — '
            'or import data from MongoDB if it was already imported before.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        dl_col, skip_col = st.columns([1, 3])
        with dl_col:
            if st.button("⬇️ Auto-Download Dataset", type="primary", use_container_width=True):
                with st.spinner("Fetching dataset from UCI …"):
                    success = _download_dataset()
                if success:
                    st.success("✅ Dataset downloaded! You can now import it.")
                    st.rerun()
                else:
                    st.error("Auto-download failed. See instructions below.")
                    st.markdown(
                        "**Manual steps:**\n"
                        "1. Go to https://archive.ics.uci.edu/dataset/352/online+retail\n"
                        "2. Download & extract `Online Retail.xlsx`\n"
                        f"3. Place it at: `{config.DATASET_PATH}`"
                    )
        with skip_col:
            # If MongoDB already has data, let user skip dataset import
            hc_quick = db.health_check()
            if hc_quick["ok"] and db.get_all_sales():
                st.markdown(
                    '<div class="custom-info">ℹ️ MongoDB already has sales data — you can use all other pages without importing the dataset again.</div>',
                    unsafe_allow_html=True,
                )
        if not dataset_exists:
            st.stop()

    section_title("Import & Manage Data")
    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("🔄 Import & Clean Dataset", type="primary", use_container_width=True):
            with st.spinner("Loading raw data …"):
                raw_df = load_raw_data(config.DATASET_PATH)
            st.session_state["raw_df"] = raw_df
            st.session_state["raw_summary"] = dataset_summary(raw_df)

            with st.spinner("Cleaning transactions …"):
                clean_df = clean_transactions(raw_df)
            st.session_state["clean_df"] = clean_df
            st.session_state["clean_sum"] = cleaning_summary(raw_df, clean_df)

            with st.spinner("Aggregating to daily revenue …"):
                daily_df = aggregate_daily_revenue(clean_df)
            st.session_state["daily_df"] = daily_df

            hc = db.health_check()
            if hc["ok"]:
                db.init_db()
                records = daily_df.copy()
                records["date"] = records["date"].astype(str)
                result = db.insert_sales_records(records.to_dict("records"))
                st.success(f"✅ MongoDB: {result['inserted']} inserted, {result['updated']} updated.")
            else:
                st.warning("MongoDB offline — data loaded in memory only.")
            st.success("Import complete!")

        if st.button("🗑️ Clear MongoDB Sales Data", use_container_width=True):
            if db.health_check()["ok"]:
                db.get_db()[config.SALES_COLLECTION].delete_many({})
                st.session_state.pop("daily_df", None)
                st.success("Sales collection cleared.")

    with col2:
        if ss_get("raw_summary"):
            s = ss_get("raw_summary")
            section_title("Raw Dataset Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Rows", f"{s['total_rows']:,}")
            c2.metric("Date Range", f"{s['date_min']} → {s['date_max']}")
            c3.metric("Unique Products", f"{s['unique_products']:,}")

    if ss_get("clean_sum"):
        cs = ss_get("clean_sum")
        section_title("Cleaning Summary")
        cols = st.columns(4)
        for col, lbl, val in zip(
            cols,
            ["Raw Rows", "Clean Rows", "Removed", "Retention %"],
            [f"{cs['raw_rows']:,}", f"{cs['clean_rows']:,}", f"{cs['removed_rows']:,}", f"{cs['retention_pct']}%"],
        ):
            col.metric(lbl, val)

    if ss_get("daily_df") is not None:
        daily_df = ss_get("daily_df")
        section_title("Daily Revenue Preview")

        fc1, fc2 = st.columns(2)
        min_d = daily_df["date"].min().date()
        max_d = daily_df["date"].max().date()
        start_d = fc1.date_input("From", min_d, min_value=min_d, max_value=max_d)
        end_d   = fc2.date_input("To",   max_d, min_value=min_d, max_value=max_d)

        mask = (daily_df["date"].dt.date >= start_d) & (daily_df["date"].dt.date <= end_d)
        filtered = daily_df[mask].copy()
        filtered["date"] = filtered["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(filtered.head(50), use_container_width=True)
        st.caption(f"Showing up to 50 of {len(filtered):,} records in selected range.")


# ═════════════════════════════════════════════════════════════════════════════
# C. MODEL TRAINING & EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Training":
    page_pill("🤖 Model Training & Evaluation")

    daily_df = _load_daily_df()
    if daily_df is None:
        st.markdown(
            '<div class="custom-warning">⚠️ Please import data in <strong>📂 Dataset Management</strong> first.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    feat_df = build_features(daily_df)
    train_df, val_df, test_df = chronological_split(feat_df)
    feature_cols = get_feature_columns(feat_df)

    s1, s2, s3 = st.columns(3)
    for col, label, val, cls in [
        (s1, "Training Days",   len(train_df), "card-indigo"),
        (s2, "Validation Days", len(val_df),   "card-teal"),
        (s3, "Test Days",       len(test_df),  "card-amber"),
    ]:
        col.markdown(
            f'<div class="kpi-card {cls}">'
            f'<div class="kpi-value">{val:,}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    tab1, tab2 = st.tabs(["📉 Regression Models", "🔵 Classification Models"])

    with tab1:
        section_title("Train Regression Models")
        if st.button("▶ Run Regression Training", type="primary"):
            with st.spinner("Training … this may take a minute."):
                reg_metrics = run_all_regression(train_df, val_df, test_df, feature_cols)
                save_regression_metrics_csv(reg_metrics, config.REGRESSION_METRICS_CSV)
                plot_model_comparison(reg_metrics, config.MODEL_COMPARISON_PNG)
                plot_sales_trend(daily_df, config.SALES_TREND_PNG)
                st.session_state["reg_metrics"] = reg_metrics

                if db.health_check()["ok"]:
                    for m in reg_metrics:
                        db.save_model_metrics({
                            "model_name": m["model"], "task_type": "regression",
                            "r2": m["r2"], "mae": m["mae"], "version": "1.0",
                        })

            st.success("✅ Regression models trained and saved.")

        if ss_get("reg_metrics"):
            metrics_df = pd.DataFrame(ss_get("reg_metrics"))
            section_title("Results")
            st.dataframe(
                metrics_df.style.highlight_max("r2", color="#1e3a5f").highlight_min("mae", color="#1e3a5f"),
                use_container_width=True,
            )
            if os.path.exists(config.MODEL_COMPARISON_PNG):
                st.image(config.MODEL_COMPARISON_PNG, caption="Model Comparison", use_container_width=True)

        elif os.path.exists(config.REGRESSION_METRICS_CSV):
            section_title("Saved Regression Metrics")
            st.dataframe(pd.read_csv(config.REGRESSION_METRICS_CSV), use_container_width=True)
            if os.path.exists(config.MODEL_COMPARISON_PNG):
                st.image(config.MODEL_COMPARISON_PNG, caption="Model Comparison", use_container_width=True)

    with tab2:
        section_title("Train Classification Models")
        if st.button("▶ Run Classification Training", type="primary"):
            with st.spinner("Training classifiers …"):
                train_clf, threshold = add_classification_target(train_df, train_df)
                val_clf, _   = add_classification_target(val_df, train_df)
                test_clf, _  = add_classification_target(test_df, train_df)

                clf_metrics = run_all_classification(train_clf, val_clf, test_clf, feature_cols)
                save_classification_metrics_csv(clf_metrics, config.CLASSIFICATION_METRICS_CSV)
                st.session_state["clf_metrics"] = clf_metrics
                st.session_state["clf_threshold"] = threshold

                if db.health_check()["ok"]:
                    for m in clf_metrics:
                        db.save_model_metrics({
                            "model_name": m["model"], "task_type": "classification",
                            "accuracy": m["accuracy"], "precision": m["precision"],
                            "recall": m["recall"], "f1": m["f1"],
                            "threshold": threshold, "version": "1.0",
                        })

            st.success(f"✅ Classifiers trained. High-sales threshold: £{threshold:,.0f}/day")

        if ss_get("clf_metrics"):
            st.dataframe(pd.DataFrame(ss_get("clf_metrics")), use_container_width=True)
            if os.path.exists(config.CONFUSION_MATRIX_PNG):
                st.image(config.CONFUSION_MATRIX_PNG, caption="Confusion Matrix — Random Forest", use_container_width=True)

        elif os.path.exists(config.CLASSIFICATION_METRICS_CSV):
            st.dataframe(pd.read_csv(config.CLASSIFICATION_METRICS_CSV), use_container_width=True)
            if os.path.exists(config.CONFUSION_MATRIX_PNG):
                st.image(config.CONFUSION_MATRIX_PNG, caption="Confusion Matrix — Random Forest", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# D. SALES FORECASTING
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Sales Forecasting":
    page_pill("🔮 Sales Forecasting")

    daily_df = _load_daily_df()
    if daily_df is None:
        st.markdown(
            '<div class="custom-warning">⚠️ Please import data first.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    col1, col2 = st.columns([1, 3])
    with col1:
        section_title("Settings")
        horizon = st.slider("Forecast horizon (days)", 7, 90, 30, help="How many future days to forecast")
        run_btn = st.button("🔮 Generate Forecast", type="primary", use_container_width=True)

    with col2:
        if run_btn:
            try:
                with st.spinner(f"Forecasting next {horizon} days …"):
                    forecast_df = generate_forecast(daily_df, horizon_days=horizon)
                    save_forecast_csv(forecast_df, config.FORECAST_CSV)
                    st.session_state["forecast_df"] = forecast_df

                    if db.health_check()["ok"]:
                        preds = forecast_df.copy()
                        preds["forecast_date"] = preds["date"].astype(str)
                        db.save_predictions(preds.to_dict("records"))

                st.success("✅ Forecast generated and saved.")
            except FileNotFoundError as e:
                st.error(str(e))

    forecast_df = ss_get("forecast_df")
    if forecast_df is None and os.path.exists(config.FORECAST_CSV):
        forecast_df = pd.read_csv(config.FORECAST_CSV)
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    if forecast_df is not None:
        section_title("Forecast Chart")
        hist_tail = daily_df.tail(90).copy()

        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_tail["date"], y=hist_tail["revenue"],
                mode="lines", name="Historical",
                line=dict(color="#818cf8", width=2),
                hovertemplate="<b>%{x|%b %d}</b><br>Historical: £%{y:,.0f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df["date"], y=forecast_df["predicted_revenue"],
                mode="lines", name="Forecast",
                line=dict(color="#f472b6", width=2.5, dash="dot"),
                fill="tozeroy", fillcolor="rgba(244,114,182,0.07)",
                hovertemplate="<b>%{x|%b %d}</b><br>Forecast: £%{y:,.0f}<extra></extra>",
            ))
            fig.add_vline(
                x=str(hist_tail["date"].max()),
                line_dash="dash", line_color="rgba(148,163,184,0.4)",
                annotation_text="Forecast Start",
                annotation_font_color="#94a3b8",
                annotation_font_size=11,
            )
            fig.update_layout(
                **PLOTLY_LAYOUT, height=380,
                xaxis_title="Date", yaxis_title="Revenue (£)",
                title=dict(text=f"Revenue Forecast — Next {len(forecast_df)} Days",
                           font=dict(size=14, color="#e2e8f0")),
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(hist_tail["date"], hist_tail["revenue"], color="#818cf8", linewidth=1.5, label="Historical")
            ax.plot(forecast_df["date"], forecast_df["predicted_revenue"], color="#f472b6",
                    linestyle="--", linewidth=2, label="Forecast")
            ax.fill_between(forecast_df["date"], forecast_df["predicted_revenue"], alpha=0.1, color="#f472b6")
            ax.axvline(x=hist_tail["date"].max(), color="#475569", linestyle=":", alpha=0.7, label="Forecast start")
            ax.legend(); ax.set_xlabel("Date"); ax.set_ylabel("Revenue (£)")
            ax.set_facecolor("#0d0f22"); fig.patch.set_facecolor("#0d0f22")
            ax.tick_params(colors="#94a3b8"); ax.grid(axis="y", linestyle="--", alpha=0.2, color="#334155")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        section_title("Forecast Summary")
        s1, s2, s3 = st.columns(3)
        avg_rev = forecast_df["predicted_revenue"].mean()
        max_rev = forecast_df["predicted_revenue"].max()
        total_fc_rev = forecast_df["predicted_revenue"].sum()
        s1.metric("Avg Daily Revenue", f"£{avg_rev:,.0f}")
        s2.metric("Peak Day Revenue", f"£{max_rev:,.0f}")
        s3.metric("Total Forecast Revenue", f"£{total_fc_rev:,.0f}")

        section_title("Forecast Table")
        display_fc = forecast_df.copy()
        display_fc["date"] = pd.to_datetime(display_fc["date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(display_fc[["date", "predicted_revenue", "model_name"]], use_container_width=True)

        csv_data = display_fc.to_csv(index=False)
        st.download_button("⬇️ Download Forecast CSV", csv_data,
                           file_name="forecast_predictions.csv", mime="text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# E. ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📈 Analytics":
    page_pill("📈 Sales Analytics")

    daily_df = _load_daily_df()
    if daily_df is None:
        st.markdown(
            '<div class="custom-warning">⚠️ Please import data first.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    section_title("Monthly Revenue")
    monthly = monthly_revenue_from_df(daily_df)

    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=monthly["year_month"],
            y=monthly["monthly_revenue"],
            marker=dict(
                color=monthly["monthly_revenue"],
                colorscale=[[0, "#312e81"], [0.5, "#6366f1"], [1, "#a5b4fc"]],
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            hovertemplate="<b>%{x}</b><br>Revenue: £%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title="Month", yaxis_title="Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(monthly["year_month"], monthly["monthly_revenue"], color="#6366f1")
        ax.set_facecolor("#0d0f22"); fig.patch.set_facecolor("#0d0f22")
        ax.tick_params(colors="#94a3b8"); plt.xticks(rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.2, color="#334155")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.dataframe(monthly, use_container_width=True)

    if ss_get("clean_df") is not None:
        section_title("Top 10 Products by Revenue")
        top = top_products_from_df(ss_get("clean_df"))
        try:
            import plotly.graph_objects as go
            fig2 = go.Figure(go.Bar(
                x=top["total_revenue"],
                y=top["Description"].str[:45],
                orientation="h",
                marker=dict(
                    color=top["total_revenue"],
                    colorscale=[[0, "#831843"], [0.5, "#ec4899"], [1, "#f9a8d4"]],
                ),
                hovertemplate="<b>%{y}</b><br>Revenue: £%{x:,.0f}<extra></extra>",
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_title="Total Revenue (£)")
            fig2.update_yaxes(autorange="reversed")
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.barh(top["Description"].str[:40], top["total_revenue"], color="#ec4899")
            ax2.invert_yaxis(); ax2.set_facecolor("#0d0f22"); fig2.patch.set_facecolor("#0d0f22")
            ax2.tick_params(colors="#94a3b8"); plt.tight_layout(); st.pyplot(fig2); plt.close()
        st.dataframe(top, use_container_width=True)
    else:
        st.markdown(
            '<div class="custom-info">ℹ️ Import data to see top product analysis.</div>',
            unsafe_allow_html=True,
        )

    section_title("Daily Revenue Distribution")
    try:
        import plotly.figure_factory as ff
        hist_data = [daily_df["revenue"].dropna().tolist()]
        fig3 = ff.create_distplot(
            hist_data, ["Revenue"],
            colors=["#14b8a6"],
            bin_size=(daily_df["revenue"].max() - daily_df["revenue"].min()) / 40,
            show_rug=False,
        )
        fig3.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="Daily Revenue (£)", yaxis_title="Density")
        st.plotly_chart(fig3, use_container_width=True)
    except Exception:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.hist(daily_df["revenue"], bins=40, color="#14b8a6", edgecolor="#0d0f22", linewidth=0.5)
        ax3.set_facecolor("#0d0f22"); fig3.patch.set_facecolor("#0d0f22")
        ax3.tick_params(colors="#94a3b8"); ax3.grid(axis="y", linestyle="--", alpha=0.2, color="#334155")
        plt.tight_layout(); st.pyplot(fig3); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# F. PREDICTION HISTORY
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📜 Prediction History":
    page_pill("📜 Prediction History")

    hc = db.health_check()
    if not hc["ok"]:
        st.error("MongoDB is offline. Cannot retrieve prediction history.")
        st.stop()

    preds = db.get_predictions(limit=500)
    if not preds:
        st.markdown(
            '<div class="custom-info">ℹ️ No predictions saved yet. Generate a forecast in <strong>🔮 Sales Forecasting</strong>.</div>',
            unsafe_allow_html=True,
        )
    else:
        df = pd.DataFrame(preds)

        section_title("Filter & Explore")
        f1, f2 = st.columns([2, 4])
        with f1:
            models = df["model_name"].unique().tolist()
            sel_model = st.selectbox("Filter by model", ["All"] + models)
        if sel_model != "All":
            df = df[df["model_name"] == sel_model]

        k1, k2, k3 = st.columns(3)
        k1.metric("Records", len(df))
        if "predicted_revenue" in df.columns:
            k2.metric("Avg Predicted Revenue", f"£{df['predicted_revenue'].mean():,.0f}")
            k3.metric("Total Predicted Revenue", f"£{df['predicted_revenue'].sum():,.0f}")

        section_title("Records")
        show_cols = [c for c in ["forecast_date", "predicted_revenue", "model_name",
                                  "horizon_days", "prediction_ts"] if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True)

        csv_data = df[show_cols].to_csv(index=False)
        st.download_button("⬇️ Download as CSV", csv_data,
                           file_name="prediction_history.csv", mime="text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# G. PROJECT INFO
# ═════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ Project Info":
    page_pill("ℹ️ Project Information")

    o1, o2 = st.columns(2)
    with o1:
        section_title("Dataset")
        st.markdown("""
- **Source**: UCI Online Retail Dataset
- **License**: CC BY 4.0
- **Period**: Dec 2010 – Dec 2011
- **Size**: ~541,000 transactions
- **Retailer**: UK-based online store
        """)

    with o2:
        section_title("Tech Stack")
        st.markdown("""
- **Database**: MongoDB + PyMongo
- **ML**: Scikit-learn (LR, RF, DT)
- **Frontend**: Streamlit
- **Data**: Pandas, NumPy
- **Viz**: Plotly, Matplotlib, Seaborn
        """)

    section_title("Database Collections")
    st.markdown("""
<table class="info-table">
<thead><tr><th>Collection</th><th>Purpose</th></tr></thead>
<tbody>
<tr><td>sales</td><td>Cleaned daily aggregated revenue records</td></tr>
<tr><td>predictions</td><td>Forecast outputs per model run</td></tr>
<tr><td>model_metrics</td><td>Evaluation results (R², MAE, accuracy, F1)</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)

    section_title("ML Models")
    st.markdown("""
<table class="info-table">
<thead><tr><th>Task</th><th>Models</th></tr></thead>
<tbody>
<tr><td>Regression</td><td>Baseline (Lag-1) · Linear Regression · Random Forest</td></tr>
<tr><td>Classification</td><td>Logistic Regression · Decision Tree · Random Forest</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)

    section_title("Features Used")
    st.markdown("""
<table class="info-table">
<thead><tr><th>Category</th><th>Features</th></tr></thead>
<tbody>
<tr><td>Calendar</td><td>day_of_week · month · quarter · week_of_year · is_weekend</td></tr>
<tr><td>Lag</td><td>lag_1 · lag_7 · lag_14</td></tr>
<tr><td>Rolling</td><td>rolling_mean_7 · rolling_mean_14</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)

    section_title("How to Run")
    st.code("""# 1. Activate virtual environment
.venv\\Scripts\\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place dataset at data/Online Retail.xlsx

# 4. Run full pipeline
python main.py

# 5. Launch dashboard
streamlit run app.py

# 6. Run tests
pytest -q

# 7. Generate PDF report
python reports/generate_report.py""", language="powershell")

    section_title("Limitations")
    st.markdown("""
<div class="custom-warning">
⚠️ Forecasts are estimates, not guarantees. The dataset covers a single UK retailer in 2010–2011;
results may not generalise. Classification threshold is median-based and may shift with different data periods.
</div>
""", unsafe_allow_html=True)
