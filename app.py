"""
app.py — Streamlit Dashboard
E-commerce Sales Prediction & Analytics System

Sections:
  0. Home Page
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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — MATRIX theme: premium black × neon green
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

/* ── Pure black app background ── */
.stApp {
    background: #000000;
    background-attachment: fixed;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #080808 !important;
    border-right: 1px solid rgba(0,255,136,0.12) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.88rem !important;
    padding: 0.6rem 1rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    cursor: pointer;
    color: #6b7280 !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0,255,136,0.06) !important;
    color: #00ff88 !important;
}
section[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"]:has(input:checked) {
    background: rgba(0,255,136,0.08) !important;
    color: #00ff88 !important;
    border-left: 2px solid #00ff88 !important;
}

/* ── Home page hero ── */
.home-hero {
    background: #000000;
    border: 1px solid rgba(0,255,136,0.1);
    border-radius: 20px;
    padding: 3.5rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.home-hero::before {
    content: '';
    position: absolute;
    top: -120px; right: -80px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(0,255,136,0.06) 0%, transparent 65%);
    pointer-events: none;
    animation: glow-pulse 4s ease-in-out infinite alternate;
}
.home-hero::after {
    content: '';
    position: absolute;
    bottom: -100px; left: -60px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,200,83,0.04) 0%, transparent 65%);
    pointer-events: none;
    animation: glow-pulse 6s ease-in-out infinite alternate-reverse;
}
@keyframes glow-pulse {
    0%   { opacity: 0.5; transform: scale(1); }
    100% { opacity: 1;   transform: scale(1.2); }
}
.home-logo {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.3em;
    color: #00ff88;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.home-logo::before {
    content: '';
    display: inline-block;
    width: 8px; height: 8px;
    background: #00ff88;
    border-radius: 50%;
    box-shadow: 0 0 10px #00ff88, 0 0 20px #00ff88;
    animation: dot-blink 2s ease-in-out infinite;
}
@keyframes dot-blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.home-title {
    font-size: 3.2rem;
    font-weight: 900;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 1rem;
    letter-spacing: -1.5px;
}
.home-title span {
    color: #00ff88;
    text-shadow: 0 0 30px rgba(0,255,136,0.5);
}
.home-subtitle {
    font-size: 1.05rem;
    color: #6b7280;
    max-width: 600px;
    line-height: 1.7;
    margin-bottom: 2rem;
    font-weight: 400;
}
.home-cta {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #00ff88;
    color: #000000;
    font-weight: 700;
    font-size: 0.9rem;
    padding: 0.75rem 1.8rem;
    border-radius: 8px;
    text-decoration: none;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
    box-shadow: 0 0 20px rgba(0,255,136,0.3);
}
.home-cta:hover {
    background: #00e57a;
    box-shadow: 0 0 40px rgba(0,255,136,0.5);
    transform: translateY(-1px);
}

/* ── Feature cards (home page) ── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-top: 2.5rem;
}
.feature-card {
    background: #0a0a0a;
    border: 1px solid rgba(0,255,136,0.08);
    border-radius: 14px;
    padding: 1.5rem 1.4rem;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00ff88, transparent);
    opacity: 0;
    transition: opacity 0.3s ease;
}
.feature-card:hover {
    border-color: rgba(0,255,136,0.25);
    background: #0d0d0d;
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,255,136,0.08);
}
.feature-card:hover::before {
    opacity: 1;
}
.feature-icon {
    font-size: 1.6rem;
    margin-bottom: 0.8rem;
    display: block;
}
.feature-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.4rem;
    letter-spacing: 0.01em;
}
.feature-desc {
    font-size: 0.77rem;
    color: #4b5563;
    line-height: 1.5;
}

/* ── Stats row (home page) ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}
.stat-tile {
    background: #0a0a0a;
    border: 1px solid rgba(0,255,136,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.stat-number {
    font-size: 1.6rem;
    font-weight: 800;
    color: #00ff88;
    font-family: 'JetBrains Mono', monospace;
    text-shadow: 0 0 15px rgba(0,255,136,0.4);
    line-height: 1;
}
.stat-label {
    font-size: 0.75rem;
    color: #4b5563;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Hero banner (inner pages) ── */
.hero-banner {
    background: #080808;
    border: 1px solid rgba(0,255,136,0.1);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    border-left: 3px solid #00ff88;
}
.hero-banner::after {
    content: '';
    position: absolute;
    top: -60%; right: -5%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,255,136,0.04) 0%, transparent 65%);
    pointer-events: none;
}
.hero-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.hero-title span { color: #00ff88; }
.hero-subtitle {
    font-size: 0.9rem;
    color: #4b5563;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.2rem 0.65rem;
    border-radius: 4px;
    margin-right: 0.4rem;
    margin-top: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.hero-badge.b1 { background: rgba(0,255,136,0.1); border: 1px solid rgba(0,255,136,0.3); color: #00ff88; }
.hero-badge.b2 { background: rgba(0,200,83,0.1);  border: 1px solid rgba(0,200,83,0.3);  color: #00c853; }
.hero-badge.b3 { background: rgba(0,230,118,0.08);border: 1px solid rgba(0,230,118,0.25);color: #69f0ae; }
.hero-badge.b4 { background: rgba(0,255,136,0.06);border: 1px solid rgba(0,255,136,0.2); color: #b9f6ca; }

/* ── KPI cards ── */
.kpi-card {
    background: #0a0a0a;
    border: 1px solid rgba(0,255,136,0.1);
    border-radius: 14px;
    padding: 1.5rem 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.card-g1::after { background: #00ff88; box-shadow: 0 0 10px #00ff88; }
.kpi-card.card-g2::after { background: #00e676; box-shadow: 0 0 10px #00e676; }
.kpi-card.card-g3::after { background: #69f0ae; box-shadow: 0 0 10px #69f0ae; }
.kpi-card.card-g4::after { background: #b9f6ca; box-shadow: 0 0 10px #b9f6ca; }
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0,255,136,0.25);
    box-shadow: 0 10px 40px rgba(0,255,136,0.08);
}
.kpi-icon { font-size: 1.5rem; margin-bottom: 0.55rem; }
.kpi-value {
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
    color: #00ff88;
    text-shadow: 0 0 20px rgba(0,255,136,0.4);
}
.kpi-label {
    font-size: 0.7rem;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
}

/* ── Section titles ── */
.section-title {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    font-size: 0.92rem;
    font-weight: 700;
    color: #d1d5db;
    margin: 2rem 0 1rem;
    padding-bottom: 0.55rem;
    border-bottom: 1px solid rgba(0,255,136,0.08);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.section-title .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 8px #00ff88;
    flex-shrink: 0;
}

/* ── Page pill ── */
.page-pill {
    display: inline-flex;
    align-items: center;
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.2);
    color: #00ff88;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.3rem 0.9rem;
    border-radius: 4px;
    margin-bottom: 1.2rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Status badges ── */
.status-online {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.25);
    color: #00ff88;
    font-size: 0.76rem; font-weight: 600;
    padding: 0.3rem 0.75rem; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
}
.status-offline {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.25);
    color: #f87171;
    font-size: 0.76rem; font-weight: 600;
    padding: 0.3rem 0.75rem; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    font-size: 1.3rem;
    font-weight: 900;
    color: #00ff88;
    letter-spacing: -0.5px;
    text-shadow: 0 0 20px rgba(0,255,136,0.5);
    margin-bottom: 0.1rem;
}
.sidebar-tagline {
    font-size: 0.68rem;
    color: #374151;
    margin-bottom: 1.4rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(0,255,136,0.08);
    margin: 1rem 0;
}

/* ── Alert boxes ── */
.custom-info {
    background: rgba(0,255,136,0.04);
    border: 1px solid rgba(0,255,136,0.15);
    border-left: 3px solid #00ff88;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    font-size: 0.86rem;
    color: #6b7280;
}
.custom-success {
    background: rgba(0,200,83,0.05);
    border: 1px solid rgba(0,200,83,0.2);
    border-left: 3px solid #00c853;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    font-size: 0.86rem;
    color: #69f0ae;
}
.custom-warning {
    background: rgba(245,158,11,0.05);
    border: 1px solid rgba(245,158,11,0.2);
    border-left: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    font-size: 0.86rem;
    color: #fcd34d;
}

/* ── Streamlit overrides ── */
.stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid rgba(0,255,136,0.08) !important; }
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"] {
    background: #00ff88 !important;
    border: none !important;
    color: #000000 !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #00e57a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 30px rgba(0,255,136,0.4) !important;
}
.stButton > button:not([kind="primary"]) {
    background: #111111 !important;
    border: 1px solid rgba(0,255,136,0.15) !important;
    color: #9ca3af !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: rgba(0,255,136,0.35) !important;
    color: #00ff88 !important;
    background: rgba(0,255,136,0.05) !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #00ff88 !important;
    text-shadow: 0 0 15px rgba(0,255,136,0.3) !important;
}
div[data-testid="stMetricLabel"] { color: #4b5563 !important; font-size: 0.75rem !important; }
.stTabs [data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    color: #4b5563 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #00ff88 !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #080808 !important;
    border-radius: 8px !important;
    border: 1px solid rgba(0,255,136,0.08) !important;
    gap: 0.25rem !important;
    padding: 0.25rem !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background: rgba(0,255,136,0.1) !important;
    border-radius: 6px !important;
}
.stSelectbox > div > div,
.stSlider > div {
    background: #0a0a0a !important;
    border-color: rgba(0,255,136,0.15) !important;
    border-radius: 8px !important;
}
.stProgress > div > div { background: #00ff88 !important; border-radius: 99px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #000000; }
::-webkit-scrollbar-thumb {
    background: rgba(0,255,136,0.3);
    border-radius: 2px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(0,255,136,0.6); }

/* ── Info table ── */
.info-table {
    width: 100%; border-collapse: collapse;
    margin: 1rem 0; font-size: 0.86rem;
}
.info-table th {
    background: rgba(0,255,136,0.06);
    color: #00ff88;
    padding: 0.65rem 1rem;
    text-align: left;
    font-weight: 700;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid rgba(0,255,136,0.1);
}
.info-table td {
    padding: 0.6rem 1rem;
    border-bottom: 1px solid rgba(0,255,136,0.04);
    color: #6b7280;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}
.info-table tr:hover td {
    background: rgba(0,255,136,0.02);
    color: #9ca3af;
}
.info-table tr:last-child td { border-bottom: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly MATRIX theme
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,8,8,0.95)",
    font=dict(family="Inter", color="#4b5563", size=11),
    xaxis=dict(
        gridcolor="rgba(0,255,136,0.05)",
        zerolinecolor="rgba(0,255,136,0.08)",
        linecolor="rgba(0,255,136,0.08)",
        tickcolor="#374151",
        tickfont=dict(color="#4b5563"),
    ),
    yaxis=dict(
        gridcolor="rgba(0,255,136,0.05)",
        zerolinecolor="rgba(0,255,136,0.08)",
        linecolor="rgba(0,255,136,0.08)",
        tickcolor="#374151",
        tickfont=dict(color="#4b5563"),
    ),
    margin=dict(l=20, r=20, t=45, b=20),
    legend=dict(
        bgcolor="rgba(8,8,8,0.8)",
        bordercolor="rgba(0,255,136,0.15)",
        borderwidth=1,
        font=dict(size=11, color="#6b7280"),
    ),
    hoverlabel=dict(
        bgcolor="#0a0a0a",
        bordercolor="rgba(0,255,136,0.4)",
        font=dict(family="Inter", size=12, color="#00ff88"),
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
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-brand">MOX</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Analytics Platform</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Dashboard",
            "📂 Dataset",
            "🤖 Model Training",
            "🔮 Forecasting",
            "📈 Analytics",
            "📜 Prediction History",
            "ℹ️ Project Info",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    hc = db.health_check()
    if hc["ok"]:
        st.markdown('<div class="status-online">● MongoDB</div>', unsafe_allow_html=True)
        st.caption(f"`{config.DB_NAME}`")
    else:
        st.markdown('<div class="status-offline">● Offline</div>', unsafe_allow_html=True)
        st.caption(hc["message"])

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.caption(f"🕐 {datetime.now().strftime('%d %b %Y · %H:%M')}")


def section_title(label: str):
    st.markdown(
        f'<div class="section-title"><span class="dot"></span>{label}</div>',
        unsafe_allow_html=True,
    )


def page_pill(label: str):
    st.markdown(f'<div class="page-pill">{label}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 0. HOME PAGE
# ═════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    # Hero
    st.markdown("""
<div class="home-hero">
  <div class="home-logo">MOX Analytics · Live</div>
  <div class="home-title">
    E-commerce<br><span>Intelligence</span> Platform
  </div>
  <div class="home-subtitle">
    Predict revenue, train ML models, and generate sales forecasts powered by
    real retail data. Built on MongoDB, Scikit-learn, and Streamlit.
  </div>
</div>
""", unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
<div class="feature-grid">
  <div class="feature-card">
    <span class="feature-icon">📊</span>
    <div class="feature-title">Live Dashboard</div>
    <div class="feature-desc">Real-time KPIs, revenue trends, and model metric summaries at a glance.</div>
  </div>
  <div class="feature-card">
    <span class="feature-icon">🤖</span>
    <div class="feature-title">ML Model Training</div>
    <div class="feature-desc">Train regression and classification models with one click. Compare R², MAE, and F1.</div>
  </div>
  <div class="feature-card">
    <span class="feature-icon">🔮</span>
    <div class="feature-title">Sales Forecasting</div>
    <div class="feature-desc">Generate multi-day revenue forecasts. Export predictions as CSV instantly.</div>
  </div>
  <div class="feature-card">
    <span class="feature-icon">📈</span>
    <div class="feature-title">Product Analytics</div>
    <div class="feature-desc">Top product rankings, monthly revenue trends, and daily distribution analysis.</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats row
    daily_df_home = _load_daily_df()
    if daily_df_home is not None and not daily_df_home.empty:
        stats = db.get_total_stats()
        total_rev  = stats.get("total_revenue",      daily_df_home["revenue"].sum())
        total_tx   = stats.get("total_transactions", daily_df_home["num_transactions"].sum())
        record_ct  = stats.get("record_count",       len(daily_df_home))
        st.markdown(f"""
<div class="stats-row">
  <div class="stat-tile">
    <div>
      <div class="stat-number">£{total_rev/1_000_000:.1f}M</div>
      <div class="stat-label">Total Revenue</div>
    </div>
  </div>
  <div class="stat-tile">
    <div>
      <div class="stat-number">{total_tx:,.0f}</div>
      <div class="stat-label">Transactions</div>
    </div>
  </div>
  <div class="stat-tile">
    <div>
      <div class="stat-number">{record_ct:,}</div>
      <div class="stat-label">Days of Data</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="custom-info">
  ℹ️ No data loaded yet. Navigate to <strong>📂 Dataset</strong> in the sidebar to import your data and unlock all features.
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tech stack tags
    section_title("Tech Stack")
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, name, desc in [
        (c1, "MongoDB", "NoSQL Database"),
        (c2, "Scikit-learn", "Machine Learning"),
        (c3, "Streamlit", "Dashboard UI"),
        (c4, "Plotly", "Interactive Charts"),
        (c5, "Pandas / NumPy", "Data Processing"),
    ]:
        col.markdown(f"""
<div class="feature-card" style="text-align:center; padding:1rem;">
  <div class="feature-title" style="color:#00ff88;">{name}</div>
  <div class="feature-desc">{desc}</div>
</div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# A. DASHBOARD OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📊 Dashboard":
    st.markdown("""
<div class="hero-banner">
  <div class="hero-title">📊 Dashboard <span>Overview</span></div>
  <div class="hero-subtitle">Live metrics, revenue trends & model performance</div>
  <div style="margin-top:0.85rem;">
    <span class="hero-badge b1">MongoDB</span>
    <span class="hero-badge b2">Scikit-learn</span>
    <span class="hero-badge b3">Streamlit</span>
    <span class="hero-badge b4">PyMongo</span>
  </div>
</div>
""", unsafe_allow_html=True)

    hc = db.health_check()
    daily_df = _load_daily_df()

    if daily_df is None or daily_df.empty:
        st.markdown(
            '<div class="custom-info">ℹ️ No sales data found. Please go to <strong>📂 Dataset</strong> and import the dataset first.</div>',
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
            (c1, "card-g1", "💰", f"£{total_rev:,.0f}", "Total Revenue"),
            (c2, "card-g2", "📦", f"{total_qty:,.0f}", "Units Sold"),
            (c3, "card-g3", "🧾", f"{total_tx:,.0f}", "Transactions"),
            (c4, "card-g4", "📅", f"{record_ct:,}", "Days of Data"),
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
                line=dict(color="#00ff88", width=2),
                fill="tozeroy",
                fillcolor="rgba(0,255,136,0.04)",
                name="Revenue",
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>Revenue: £%{y:,.0f}<extra></extra>",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                              xaxis_title="Date", yaxis_title="Revenue (£)")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(daily_df["date"], daily_df["revenue"], color="#00ff88", linewidth=1.5)
            ax.fill_between(daily_df["date"], daily_df["revenue"], alpha=0.06, color="#00ff88")
            ax.set_facecolor("#080808"); fig.patch.set_facecolor("#080808")
            ax.tick_params(colors="#4b5563"); ax.spines[:].set_color("#111111")
            ax.grid(axis="y", linestyle="--", alpha=0.1, color="#1a1a1a")
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

elif page == "📂 Dataset":
    page_pill("📂 Dataset Management")

    dataset_exists = os.path.exists(config.DATASET_PATH)
    if dataset_exists:
        st.markdown(
            '<div class="custom-success">✅ Dataset file found and ready to import.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="custom-warning">⚠️ Dataset file not found. '
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
            '<div class="custom-warning">⚠️ Please import data in <strong>📂 Dataset</strong> first.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    feat_df = build_features(daily_df)
    train_df, val_df, test_df = chronological_split(feat_df)
    feature_cols = get_feature_columns(feat_df)

    s1, s2, s3 = st.columns(3)
    for col, label, val, cls in [
        (s1, "Training Days",   len(train_df), "card-g1"),
        (s2, "Validation Days", len(val_df),   "card-g2"),
        (s3, "Test Days",       len(test_df),  "card-g3"),
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
                metrics_df.style.highlight_max("r2", color="#0a2a1a").highlight_min("mae", color="#0a2a1a"),
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

elif page == "🔮 Forecasting":
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

        # Ensure dates are proper datetime objects for both plotly and matplotlib
        hist_tail["date"] = pd.to_datetime(hist_tail["date"])
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_tail["date"], y=hist_tail["revenue"],
                mode="lines", name="Historical",
                line=dict(color="#4b5563", width=2),
                hovertemplate="<b>%{x|%b %d}</b><br>Historical: £%{y:,.0f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df["date"], y=forecast_df["predicted_revenue"],
                mode="lines", name="Forecast",
                line=dict(color="#00ff88", width=2.5, dash="dot"),
                fill="tozeroy", fillcolor="rgba(0,255,136,0.04)",
                hovertemplate="<b>%{x|%b %d}</b><br>Forecast: £%{y:,.0f}<extra></extra>",
            ))
            # Format vline timestamp as YYYY-MM-DD to avoid ISO parsing issues
            vline_x = hist_tail["date"].max().strftime("%Y-%m-%d")
            fig.add_vline(
                x=vline_x,
                line_dash="dash", line_color="rgba(0,255,136,0.25)",
                annotation_text="Forecast Start",
                annotation_font_color="#00ff88",
                annotation_font_size=11,
            )
            fig.update_layout(
                **PLOTLY_LAYOUT, height=380,
                xaxis_title="Date", yaxis_title="Revenue (£)",
                title=dict(text=f"Revenue Forecast — Next {len(forecast_df)} Days",
                           font=dict(size=13, color="#6b7280")),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            # Matplotlib fallback — convert to proper Python datetime for compatibility
            hist_dates = pd.to_datetime(hist_tail["date"]).dt.to_pydatetime()
            fc_dates   = pd.to_datetime(forecast_df["date"]).dt.to_pydatetime()
            hist_date_max = pd.to_datetime(hist_tail["date"]).max().to_pydatetime()
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(hist_dates, hist_tail["revenue"], color="#4b5563", linewidth=1.5, label="Historical")
            ax.plot(fc_dates, forecast_df["predicted_revenue"], color="#00ff88",
                    linestyle="--", linewidth=2, label="Forecast")
            ax.fill_between(fc_dates, forecast_df["predicted_revenue"], alpha=0.05, color="#00ff88")
            ax.axvline(x=hist_date_max, color="#374151", linestyle=":", alpha=0.7, label="Forecast start")
            ax.legend(); ax.set_xlabel("Date"); ax.set_ylabel("Revenue (£)")
            ax.set_facecolor("#080808"); fig.patch.set_facecolor("#080808")
            ax.tick_params(colors="#4b5563"); ax.grid(axis="y", linestyle="--", alpha=0.1, color="#111111")
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
                colorscale=[[0, "#0a2a1a"], [0.5, "#00c853"], [1, "#00ff88"]],
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            hovertemplate="<b>%{x}</b><br>Revenue: £%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title="Month", yaxis_title="Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.bar(monthly["year_month"], monthly["monthly_revenue"], color="#00c853")
        ax.set_facecolor("#080808"); fig.patch.set_facecolor("#080808")
        ax.tick_params(colors="#4b5563"); plt.xticks(rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.1, color="#111111")
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
                    colorscale=[[0, "#052a10"], [0.5, "#00c853"], [1, "#00ff88"]],
                ),
                hovertemplate="<b>%{y}</b><br>Revenue: £%{x:,.0f}<extra></extra>",
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_title="Total Revenue (£)")
            fig2.update_yaxes(autorange="reversed")
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.barh(top["Description"].str[:40], top["total_revenue"], color="#00c853")
            ax2.invert_yaxis(); ax2.set_facecolor("#080808"); fig2.patch.set_facecolor("#080808")
            ax2.tick_params(colors="#4b5563"); plt.tight_layout(); st.pyplot(fig2); plt.close()
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
            colors=["#00ff88"],
            bin_size=(daily_df["revenue"].max() - daily_df["revenue"].min()) / 40,
            show_rug=False,
        )
        fig3.update_layout(**PLOTLY_LAYOUT, height=300, xaxis_title="Daily Revenue (£)", yaxis_title="Density")
        st.plotly_chart(fig3, use_container_width=True)
    except Exception:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.hist(daily_df["revenue"], bins=40, color="#00c853", edgecolor="#080808", linewidth=0.5)
        ax3.set_facecolor("#080808"); fig3.patch.set_facecolor("#080808")
        ax3.tick_params(colors="#4b5563"); ax3.grid(axis="y", linestyle="--", alpha=0.1, color="#111111")
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
            '<div class="custom-info">ℹ️ No predictions saved yet. Generate a forecast in <strong>🔮 Forecasting</strong>.</div>',
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
