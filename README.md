# E-commerce Sales Prediction and Analytics System
### Using MongoDB, PyMongo, and Machine Learning

A complete college project that analyses historical e-commerce sales data,
stores and retrieves it through MongoDB, trains Machine Learning models to
predict sales, and displays trends and predictions in a Streamlit dashboard.

---

## Project Structure

```
moxproject1/
├── app.py                      # Streamlit dashboard
├── main.py                     # End-to-end pipeline runner
├── config.py                   # Central configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env.example                # Environment variable template
├── data/
│   ├── download_dataset.py     # Dataset download helper
│   └── README.md
├── src/
│   ├── database.py             # All MongoDB operations (PyMongo)
│   ├── data_loader.py          # Raw dataset loading
│   ├── cleaning.py             # Data cleaning pipeline
│   ├── feature_engineering.py # Lag/rolling features, splits
│   ├── train_regression.py    # Baseline, Linear Reg, Random Forest
│   ├── train_classification.py # Logistic Reg, DT, RF classifiers
│   ├── forecasting.py          # Multi-step ahead forecast
│   ├── evaluation.py           # Metrics, charts, CSV export
│   └── analytics.py            # Top products, monthly trends
├── models/                     # Saved .joblib model files
├── results/                    # CSV metrics, PNG charts, JSON summary
├── reports/
│   ├── generate_report.py      # PDF report generator
│   └── project_report.pdf      # Generated report
└── tests/                      # Pytest test suite
    ├── test_cleaning.py
    ├── test_features.py
    ├── test_evaluation.py
    ├── test_forecasting.py
    └── test_database.py
```

---

## Prerequisites

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.11+ | https://www.python.org/ |
| MongoDB Community | 6+ | https://www.mongodb.com/try/download/community |
| MongoDB Compass | Latest | https://www.mongodb.com/try/download/compass |

---

## Quick Start (Windows PowerShell)

### Step 1 — Check MongoDB is running
```powershell
Get-Service -Name MongoDB
# If not Running:
Start-Service -Name MongoDB
```

### Step 2 — Create & activate virtual environment
```powershell
cd D:\MOX\moxproject1
python -m venv .venv
.venv\Scripts\activate
```

### Step 3 — Install dependencies
```powershell
pip install -r requirements.txt
```

### Step 4 — Download the dataset
```powershell
python data/download_dataset.py
```
If automatic download fails:
1. Go to https://archive.ics.uci.edu/dataset/352/online+retail
2. Download the ZIP and extract `Online Retail.xlsx`
3. Place it in the `data/` folder

### Step 5 — Configure environment
```powershell
copy .env.example .env
# Edit .env if your MongoDB URI is different from localhost:27017
```

### Step 6 — Run the full pipeline
```powershell
python main.py
```
This will:
- Clean and import data to MongoDB
- Train all regression and classification models
- Save evaluation metrics to `results/`
- Generate the 30-day forecast
- Save charts and CSV files

### Step 7 — Launch the dashboard
```powershell
streamlit run app.py
```
Open http://localhost:8501 in your browser.

### Step 8 — Run tests
```powershell
pytest -q
```

### Step 9 — Generate PDF report
```powershell
python reports/generate_report.py
```
Report saved to `reports/project_report.pdf`

---

## MongoDB Setup

### Local MongoDB
MongoDB runs as a Windows service and starts automatically.
Connect via MongoDB Compass: `mongodb://localhost:27017`

Collections created automatically:
- `ecommerce_sales_ml.sales`
- `ecommerce_sales_ml.predictions`
- `ecommerce_sales_ml.model_metrics`

### Optional: MongoDB Atlas (Cloud)
1. Create a free cluster at https://cloud.mongodb.com/
2. Get your connection string
3. Update `.env`:
   ```
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
   ```

---

## Dataset

**Source:** UCI Online Retail Dataset  
**URL:** https://archive.ics.uci.edu/dataset/352/online+retail  
**License:** CC BY 4.0  
**Period:** December 2010 – December 2011  
**Size:** ~541,909 transactions  

**Cleaning rules applied:**
1. Remove cancellations (InvoiceNo starts with 'C')
2. Remove rows with Quantity ≤ 0
3. Remove rows with UnitPrice ≤ 0
4. Remove rows with missing Description
5. Revenue = Quantity × UnitPrice

---

## ML Models

### Regression (Revenue Prediction)
| Model | Description |
|-------|-------------|
| Baseline (Lag-1) | Predicts today = yesterday's revenue |
| Linear Regression | With StandardScaler |
| Random Forest | 200 trees, evaluated on held-out test set |

### Classification (High vs Low Sales)
| Model | Description |
|-------|-------------|
| Logistic Regression | With StandardScaler |
| Decision Tree | max_depth=6 |
| Random Forest | 200 trees, max_depth=8 |

Threshold = training-set median daily revenue (no leakage from test set).

---

## Result Files

After running `python main.py`:

| File | Description |
|------|-------------|
| `results/regression_metrics.csv` | R² and MAE for all regression models |
| `results/classification_metrics.csv` | Accuracy, F1, etc. for classifiers |
| `results/forecast_predictions.csv` | 30-day revenue forecast |
| `results/model_comparison.png` | Bar chart comparing models |
| `results/sales_trend.png` | Historical revenue line chart |
| `results/confusion_matrix.png` | Confusion matrix (RF classifier) |
| `results/evaluation_summary.json` | All metrics in JSON format |
| `reports/project_report.pdf` | Full PDF report |

---

## Troubleshooting

### MongoDB connection refused
```powershell
# Check if service is running
Get-Service MongoDB
# Start it
Start-Service MongoDB
```

### pip install fails (Python 3.14 wheel issues)
```powershell
pip install --upgrade pip
pip install numpy==1.26.4 --prefer-binary
pip install -r requirements.txt
```

### Dataset not found
Ensure `Online Retail.xlsx` is in the `data/` folder.

### Streamlit port already in use
```powershell
streamlit run app.py --server.port 8502
```

---

## Inspecting MongoDB with Compass

1. Open MongoDB Compass
2. Connect to `mongodb://localhost:27017`
3. Navigate to `ecommerce_sales_ml` database
4. Explore collections: `sales`, `predictions`, `model_metrics`

---

## Academic Integrity Note

All metrics in this project are generated from actual model training on real data.
No metrics, charts, or results have been fabricated.
Forecasts are estimates and not guarantees of future business performance.

---

*Project: E-commerce Sales Prediction and Analytics System*  
*Technology: Python · MongoDB · Scikit-learn · Streamlit*
