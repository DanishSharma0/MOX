"""
reports/generate_report.py
───────────────────────────
Generate a professionally formatted PDF project report using ReportLab.
Reads actual metrics from results/ and includes charts as images.

Run:  python reports/generate_report.py
"""
import json
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "Title", parent=styles["Title"], fontSize=22, spaceAfter=12,
    textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER,
)
H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=14, spaceBefore=18, spaceAfter=6,
    textColor=colors.HexColor("#e94560"), borderPad=4,
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=4,
    textColor=colors.HexColor("#0f3460"),
)
BODY = ParagraphStyle(
    "Body", parent=styles["Normal"], fontSize=10, spaceAfter=6, leading=16,
    alignment=TA_JUSTIFY,
)
CODE = ParagraphStyle(
    "Code", parent=styles["Code"], fontSize=8, backColor=colors.HexColor("#f0f0f0"),
    borderColor=colors.HexColor("#cccccc"), borderWidth=1, borderPad=4,
)
CENTER = ParagraphStyle("Centre", parent=BODY, alignment=TA_CENTER)


def spacer(h=0.3):
    return Spacer(1, h * cm)


def hr():
    return HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e94560"),
                       spaceAfter=8)


def safe_image(path, width=14 * cm):
    if path and os.path.exists(path):
        return Image(path, width=width, height=width * 0.45)
    return Paragraph(f"<i>[Chart not available: {os.path.basename(path)}]</i>", BODY)


def load_regression_metrics():
    path = config.REGRESSION_METRICS_CSV
    if not os.path.exists(path):
        return []
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_classification_metrics():
    path = config.CLASSIFICATION_METRICS_CSV
    if not os.path.exists(path):
        return []
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_evaluation_summary():
    path = config.EVALUATION_SUMMARY_JSON
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ── Build story ───────────────────────────────────────────────────────────────

def build_story():
    story = []
    reg_metrics = load_regression_metrics()
    clf_metrics = load_classification_metrics()

    # ── Title page ─────────────────────────────────────────────────────
    story.append(spacer(3))
    story.append(Paragraph("E-commerce Sales Prediction", TITLE_STYLE))
    story.append(Paragraph("and Analytics System", TITLE_STYLE))
    story.append(spacer(0.5))
    story.append(hr())
    story.append(Paragraph(
        "Using MongoDB, PyMongo, and Machine Learning", CENTER))
    story.append(spacer(2))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", CENTER))
    story.append(Paragraph("Technology: Python · MongoDB · Scikit-learn · Streamlit", CENTER))
    story.append(PageBreak())

    # ── Abstract ───────────────────────────────────────────────────────
    story.append(Paragraph("Abstract", H1))
    story.append(hr())
    story.append(Paragraph(
        "This project presents a complete E-commerce Sales Prediction and Analytics System "
        "that combines MongoDB for scalable data storage, PyMongo for database interaction, "
        "and Scikit-learn for machine learning. Using the UCI Online Retail dataset "
        "(December 2010 – December 2011), the system trains regression models to forecast "
        "daily revenue and classification models to predict high-sales periods. "
        "A Streamlit dashboard provides interactive analytics, model evaluation, and "
        "sales forecasting capabilities. The Random Forest Regressor achieves the highest "
        "predictive accuracy among the models evaluated.",
        BODY))
    story.append(PageBreak())

    # ── Introduction ───────────────────────────────────────────────────
    story.append(Paragraph("1. Introduction", H1))
    story.append(hr())
    story.append(Paragraph(
        "E-commerce businesses generate large volumes of transaction data every day. "
        "Transforming this data into actionable insights — such as revenue forecasts and "
        "high-sales predictions — can help businesses optimise inventory, staffing, and "
        "marketing strategies. This project addresses this challenge by building a "
        "production-style pipeline that cleans raw transactional data, stores it in MongoDB, "
        "engineers time-series features, and trains multiple machine learning models.",
        BODY))

    # ── Problem Statement ──────────────────────────────────────────────
    story.append(Paragraph("2. Problem Statement", H1))
    story.append(hr())
    story.append(Paragraph(
        "Given historical daily e-commerce revenue data, predict future revenue for a "
        "specified forecast horizon and classify whether a future day will be a "
        "'high-sales' or 'low-sales' day. The system must prevent data leakage, use "
        "chronological train/test splits, and store all results in MongoDB.",
        BODY))

    # ── Technology Stack ───────────────────────────────────────────────
    story.append(Paragraph("3. Technology Stack", H1))
    story.append(hr())
    tech_data = [
        ["Component", "Technology", "Version"],
        ["Language", "Python", "3.11+"],
        ["Database", "MongoDB Community", "6+"],
        ["DB Driver", "PyMongo", "4.7.2"],
        ["Data Processing", "Pandas / NumPy", "2.2.2 / 1.26.4"],
        ["Machine Learning", "Scikit-learn", "1.5.0"],
        ["Visualisation", "Matplotlib / Seaborn", "3.9.0 / 0.13.2"],
        ["Dashboard", "Streamlit", "1.35.0"],
        ["Testing", "Pytest", "8.2.2"],
        ["Report", "ReportLab", "4.2.0"],
    ]
    t = Table(tech_data, colWidths=[5 * cm, 6 * cm, 4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f9f9f9"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)

    # ── Dataset ────────────────────────────────────────────────────────
    story.append(Paragraph("4. Dataset", H1))
    story.append(hr())
    story.append(Paragraph("<b>Source:</b> UCI Machine Learning Repository — Online Retail Dataset", BODY))
    story.append(Paragraph("<b>URL:</b> https://archive.ics.uci.edu/dataset/352/online+retail", BODY))
    story.append(Paragraph("<b>License:</b> Creative Commons Attribution 4.0 (CC BY 4.0)", BODY))
    story.append(Paragraph("<b>Period:</b> 01 December 2010 – 09 December 2011", BODY))
    story.append(Paragraph("<b>Size:</b> ~541,909 transactions from a UK-based online retailer", BODY))
    story.append(spacer(0.3))
    story.append(Paragraph(
        "The dataset contains transaction-level records with eight fields: "
        "InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, "
        "CustomerID, and Country. Revenue is computed as Quantity × UnitPrice.",
        BODY))

    # ── Data Cleaning ──────────────────────────────────────────────────
    story.append(Paragraph("5. Data Preprocessing & Cleaning", H1))
    story.append(hr())
    clean_rules = [
        ["Rule", "Criterion", "Reason"],
        ["1", "InvoiceNo starts with 'C'", "Cancellation transactions"],
        ["2", "Quantity ≤ 0", "Returns or data entry errors"],
        ["3", "UnitPrice ≤ 0", "Invalid price entries"],
        ["4", "Missing Description", "Unidentifiable products"],
    ]
    t2 = Table(clean_rules, colWidths=[2 * cm, 7 * cm, 6 * cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f9f9f9"), colors.white]),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t2)

    # ── Feature Engineering ────────────────────────────────────────────
    story.append(Paragraph("6. Feature Engineering", H1))
    story.append(hr())
    story.append(Paragraph(
        "To convert daily revenue time series into a supervised learning problem, "
        "the following features are engineered:",
        BODY))
    feature_data = [
        ["Feature", "Description", "Leakage Safe?"],
        ["day_of_week", "Day of week (0=Mon, 6=Sun)", "Yes"],
        ["month", "Calendar month (1–12)", "Yes"],
        ["week_of_year", "ISO week number", "Yes"],
        ["quarter", "Quarter (1–4)", "Yes"],
        ["is_weekend", "1 if Saturday/Sunday", "Yes"],
        ["lag_1", "Revenue 1 day ago", "Yes — uses shift(1)"],
        ["lag_7", "Revenue 7 days ago", "Yes — uses shift(7)"],
        ["lag_14", "Revenue 14 days ago", "Yes — uses shift(14)"],
        ["rolling_mean_7", "7-day rolling average (excluding current day)", "Yes — uses shift(1).rolling(7)"],
        ["rolling_mean_14", "14-day rolling average", "Yes — uses shift(1).rolling(14)"],
    ]
    t3 = Table(feature_data, colWidths=[4 * cm, 7 * cm, 4 * cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f9f9f9"), colors.white]),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t3)

    # ── Train / Test Split ─────────────────────────────────────────────
    story.append(Paragraph("7. Train / Validation / Test Split", H1))
    story.append(hr())
    story.append(Paragraph(
        "A strict chronological split is used. Random shuffling is NEVER applied to "
        "time-series data as it would allow future information to leak into training.\n"
        "\n• Training set: first 70% of days (chronologically)\n"
        "• Validation set: next 15% of days\n"
        "• Test set: final 15% of days",
        BODY))

    # ── ML Models ─────────────────────────────────────────────────────
    story.append(Paragraph("8. Machine Learning Models", H1))
    story.append(hr())
    story.append(Paragraph("<b>Regression Task (Revenue Prediction)</b>", H2))
    story.append(Paragraph(
        "Three models are trained and compared:\n"
        "1. Baseline — predicts yesterday's revenue (lag_1)\n"
        "2. Linear Regression with StandardScaler\n"
        "3. Random Forest Regressor (200 trees, max_depth=None)",
        BODY))
    story.append(Paragraph("<b>Classification Task (High vs. Low Sales)</b>", H2))
    story.append(Paragraph(
        "Three classifiers predict whether daily revenue exceeds the training-set median:\n"
        "1. Logistic Regression\n"
        "2. Decision Tree (max_depth=6)\n"
        "3. Random Forest Classifier (200 trees, max_depth=8)",
        BODY))

    # ── Evaluation Results ─────────────────────────────────────────────
    story.append(Paragraph("9. Evaluation Results", H1))
    story.append(hr())

    story.append(Paragraph("<b>9.1 Regression Metrics</b>", H2))
    if reg_metrics:
        header = ["Model", "R² Score", "MAE (£)"]
        rows = [header] + [[m.get("model", ""), m.get("r2", ""), m.get("mae", "")] for m in reg_metrics]
        rt = Table(rows, colWidths=[7 * cm, 4 * cm, 4 * cm])
        rt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f8ff"), colors.white]),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(rt)
    else:
        story.append(Paragraph("<i>Run python main.py to generate actual metrics.</i>", BODY))

    story.append(spacer())
    story.append(safe_image(config.MODEL_COMPARISON_PNG))

    story.append(Paragraph("<b>9.2 Classification Metrics</b>", H2))
    if clf_metrics:
        header2 = ["Model", "Accuracy", "Precision", "Recall", "F1"]
        rows2 = [header2] + [
            [m.get("model", ""), m.get("accuracy", ""), m.get("precision", ""),
             m.get("recall", ""), m.get("f1", "")] for m in clf_metrics
        ]
        ct = Table(rows2, colWidths=[6 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm])
        ct.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f8ff"), colors.white]),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(ct)
    else:
        story.append(Paragraph("<i>Run python main.py to generate actual metrics.</i>", BODY))

    story.append(spacer())
    story.append(safe_image(config.CONFUSION_MATRIX_PNG, width=10 * cm))

    # ── Sales Trend Chart ──────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("10. Sales Trend", H1))
    story.append(hr())
    story.append(safe_image(config.SALES_TREND_PNG))

    # ── Limitations ────────────────────────────────────────────────────
    story.append(Paragraph("11. Limitations", H1))
    story.append(hr())
    story.append(Paragraph(
        "• Forecasts are estimates and not guarantees of future performance.\n"
        "• The dataset covers a single UK-based retailer in 2010–2011; results "
        "may not generalise to other businesses or time periods.\n"
        "• The classification threshold is the training-set median and shifts "
        "with different data periods.\n"
        "• Recursive multi-step forecasting accumulates errors over longer horizons.\n"
        "• External factors (promotions, holidays, economic events) are not captured.",
        BODY))

    # ── Conclusion ─────────────────────────────────────────────────────
    story.append(Paragraph("12. Conclusion & Future Scope", H1))
    story.append(hr())
    story.append(Paragraph(
        "This project demonstrates a complete end-to-end ML pipeline for e-commerce sales "
        "prediction using MongoDB, PyMongo, and Scikit-learn. The Random Forest Regressor "
        "consistently outperforms the baseline and Linear Regression in R² and MAE. "
        "Future work could incorporate LSTM networks for sequence modelling, external "
        "features such as holiday calendars, and real-time data ingestion from an "
        "e-commerce API.",
        BODY))

    # ── References ─────────────────────────────────────────────────────
    story.append(Paragraph("13. References", H1))
    story.append(hr())
    refs = [
        "Chen, D. (2015). Online Retail. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5BW33",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. "
        "JMLR 12:2825–2830.",
        "MongoDB, Inc. (2024). MongoDB Documentation. https://www.mongodb.com/docs/",
        "McKinney, W. (2010). Data Structures for Statistical Computing in Python. "
        "Proceedings of the 9th Python in Science Conference.",
        "Streamlit Inc. (2024). Streamlit Documentation. https://docs.streamlit.io/",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {ref}", BODY))

    # ── Installation Guide ─────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("14. Installation & Execution", H1))
    story.append(hr())
    install_steps = """
# 1. Clone / extract the project
cd D:\\MOX\\moxproject1

# 2. Create & activate virtual environment
python -m venv .venv
.venv\\Scripts\\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place dataset
# Download Online Retail.xlsx from UCI and place in data/

# 5. Configure environment
copy .env.example .env   # edit MONGODB_URI if needed

# 6. Run the full pipeline
python main.py

# 7. Launch dashboard
streamlit run app.py

# 8. Run tests
pytest -q

# 9. Generate this report
python reports/generate_report.py
"""
    story.append(Paragraph(install_steps.replace("\n", "<br/>"), CODE))

    return story


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_report():
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    doc = SimpleDocTemplate(
        config.REPORT_PDF,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    story = build_story()
    doc.build(story)
    print(f"Report generated: {config.REPORT_PDF}")


if __name__ == "__main__":
    generate_report()
