# 📊 Portfolio Risk Detection System

An end-to-end Python-based portfolio risk analysis system that computes financial risk metrics (VaR, CVaR), analyzes portfolio behavior, and generates both interactive dashboards and Excel reports.

---

## 🚀 Overview

This project simulates a real-world **financial risk analytics pipeline** used by FP&A teams, risk analysts, and hedge funds.

It performs the complete workflow:

* Loads portfolio price data
* Calculates asset and portfolio returns
* Computes risk metrics (VaR, CVaR)
* Analyzes correlations and diversification
* Generates insights and recommendations
* Exports results to Excel
* Displays interactive dashboard

---

## 🧠 Key Features

### 📉 Risk Models

* Historical Value at Risk (VaR)
* Conditional Value at Risk (CVaR)
* Parametric VaR
* Monte Carlo Simulation

### 📊 Portfolio Analytics

* Portfolio return calculation
* Variance & covariance analysis
* Correlation matrix
* Risk contribution per asset
* Concentration analysis (HHI)

### 📈 Outputs

* 📊 Streamlit Dashboard (interactive)
* 📄 Excel Report (multi-sheet)

---

## 📂 Project Structure

```
Portfolio_Risk_Detection_System/
│
├── main.py                          # Run full pipeline
├── portfolio_risk_report.xlsx       # Generated Excel output
│
├── var_analyzer/
│   ├── data/                        # Data loading
│   ├── preprocessing/               # Returns calculation
│   ├── risk_models/                 # VaR, CVaR models
│   ├── risk_analysis/               # Insights & diagnostics
│   ├── reporting/                   # Excel export
│   ├── visualization/               # Streamlit dashboard
│   ├── utils/                       # Helpers & exceptions
│   └── validation/                  # Input validation
```

---

## ⚙️ Installation

### 1. Clone Repository

```
git clone https://github.com/your-username/Portfolio_Risk_Detection_System.git
cd Portfolio_Risk_Detection_System
```

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## ▶️ How to Run

---

### 🔹 Run Full Pipeline (Excel Report)

```
python var_analyzer/main.py
```

✅ Output:

* Console summary (VaR, CVaR, Risk Level)
* Excel file generated:

  ```
  portfolio_risk_report.xlsx
  ```

---

### 🔹 Run Dashboard (Interactive UI)

```
streamlit run var_analyzer/visualization/dashboard.py
```

✅ Output:

* Opens browser dashboard
* Shows:

  * VaR & CVaR metrics
  * Portfolio returns chart
  * Correlation heatmap
  * Risk insights & diagnostics

---




## 📊 Excel Report Includes

* Summary (VaR, CVaR, Risk Level)
* Portfolio Returns
* Asset Returns
* Correlation Matrix
* Insights & Recommendations


## 📸 Screenshots

### 📊 Dashboard
![Dashboard](assets/dashboard.png)

### 📄 Excel Summary
![Summary](assets/excel_summary.png)

### 📈 Correlation Matrix
![Correlation](assets/correlation_matrix.png)

---

## 📌 Key Concepts Used

* Variance & Covariance
* Correlation
* Portfolio Diversification
* Tail Risk (CVaR)
* Monte Carlo Simulation

---

## 💡 Insights Generated

* Top Risk Contributing Asset
* Highest Correlation Pair
* Lowest Correlation Pair
* Portfolio Diversification Status
* Risk Level Classification

---

## ⚠️ Notes

* Portfolio weights must sum to 1
* Valid price data required
* Supported confidence levels: 95%, 99%

---

## 👤 Author

**Tanish Khandelwal**

* Finance + Tech
* Financial Modeling | Risk Analytics | Python

---

## ⭐ Future Improvements

* Live market data integration
* Portfolio optimization (Sharpe Ratio)
* Backtesting module
* Web deployment

---

## 🧹 Important

Do NOT upload these files to GitHub:

```
__pycache__/
*.pyc
venv/
```

Add them to `.gitignore`
