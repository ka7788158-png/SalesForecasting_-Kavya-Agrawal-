# 📊 SalesForecasting_[Kavya Agrawal]

**Internship Project — Week 3 & Week 4**
**Project Title:** End-to-End Sales Forecasting & Demand Intelligence System
**Assigned:** 03/07/2026 

---

## 🎯 Problem Statement

Every retail and e-commerce company lives and dies by one question: *how much of each product will we sell next month, and will we have enough stock to meet that demand?* This project builds an intelligent sales forecasting system that predicts future product demand, detects unusual sales spikes/drops, segments products by demand pattern, and presents everything through a deployed interactive dashboard.

The project covers **Time Series Analysis, Machine Learning, Forecasting, Anomaly Detection, Product Segmentation, and Deployment**.

---

## 📦 Dataset

- **Primary:** [Superstore Sales Dataset](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) (`train.csv`) — 4 years of daily sales data across multiple product categories and regions

---

## 🗂️ Project Structure
SalesForecasting_[Kavya Agrawal]/
- **analysis.ipynb**          # Full analysis notebook (Tasks 1–6)
- **train.csv**                # Superstore sales dataset
- **app.py**                   # Streamlit dashboard (Task 7)
- **requirements.txt**         # Python dependencies
- **summary.pdf**               # 2-page executive business report (Task 8)
- **charts/**                  # All exported chart images (.png)

---

## ✅ Tasks Completed

| Task | Description |
|------|-------------|
| 1 | Data loading, cleaning, time-feature extraction, weekly/monthly aggregation, exploratory Q&A |
| 2 | Time series decomposition (trend/seasonal/residual) + ADF stationarity testing |
| 3 | Forecasting with SARIMA, Prophet, and XGBoost — compared on MAE/RMSE/MAPE |
| 4 | Segment-level (category & region) forecasting using the best model |
| 5 | Anomaly detection using Isolation Forest and Z-Score methods |
| 6 | Product demand segmentation via K-Means clustering + PCA |
| 7 | Interactive Streamlit dashboard (Sales Overview, Forecast Explorer, Anomaly Report, Segments) |
| 8 | Executive business report for Head of Supply Chain / CFO |

---

## 🔑 Key Findings

- **Highest revenue category:** Technology (~$827,455.87 in total sales)
- **Most consistent regional growth:** East region — steady YoY growth of ~16.5%–20%
- **Average order-to-ship time:** 3.96 days overall (East fastest at 3.91 days, Central slowest at 4.06 days)
- **Seasonality:** Strong, consistent Q4 spikes (Sep, Nov, Dec) driven by holiday shopping
- **Stationarity:** ADF test on monthly sales returned p = 0.0003 → data is stationary, though strong seasonality is still modeled explicitly

### Model Comparison

| Model | MAPE | RMSE | Notes |
|-------|------|------|-------|
| SARIMA | 23.94% | 12506.52 | Statistical baseline, seasonal order (1,1,1,12) |
| Prophet | 14.48% | 7272.00 | Industry-standard, strong trend/seasonality decomposition |
| **XGBoost** | **0.12%** | **82.31** | **Recommended for production** — lag + rolling-mean features captured demand patterns far more precisely |

### Anomaly Detection

- **Isolation Forest** flagged 11 anomalies (Q4 spikes, post-holiday Q1 drops)
- **Z-Score method** (rolling 4-week window) flagged 0 anomalies — spikes build up gradually enough to avoid short-window Z-score thresholds
- Disagreement between methods highlights that *absolute* deviation ≠ *local* deviation

### Product Demand Segments (K-Means, K=4)

| Cluster | Profile | Stocking Strategy |
|---------|---------|--------------------|
| High Volume, Stable Demand | Phones, Chairs | Never stock out — automated reordering, high buffer stock |
| Low Volume, High Volatility | Copiers, Machines | Just-in-Time inventory — minimize capital tied up |
| Growing Demand | Envelopes, Fasteners, Labels | Aggressive scaling & bundling |
| Declining/Stagnant Demand | Art, Paper, Appliances | Phase out & discount, reallocate capital |

---

## 🛠️ Tools & Libraries

Python 3.x · Pandas & NumPy · Statsmodels · Prophet · XGBoost · Scikit-learn · Matplotlib/Seaborn · Streamlit · Git & GitHub

---

## 🚀 Running the Project

**1. Clone the repo & install dependencies**
```bash
git clone https://github.com/ka7788158-png/SalesForecasting_-Kavya-Agrawal-.git
cd SalesForecasting_-Kavya Agrawal-
pip install -r requirements.txt
```

**2. Run the notebook**
```bash
jupyter notebook analysis.ipynb
```

**3. Run the Streamlit dashboard locally**
```bash
streamlit run app.py
```

---

## ⚠️ Known Limitation

Forecasts are trained on 4 years of historical data and don't yet account for external demand drivers (promotions, pricing changes, macroeconomic shifts), so accuracy may degrade during unusual market conditions not reflected in the training window.

---

## 👤 Author

**[Kavya Agrawal]**
[GitHub](https://github.com/ka7788158-png) · [LinkedIn](https://linkedin.com/in/kavya-agrawal-12b289326)
