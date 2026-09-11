**Try InventoryAI live:**
 https://pradeep-inventoryai.streamlit.app/

InventoryAI is publicly deployed using Streamlit Community Cloud.

# InventoryAI 🛒

## AI-Powered Retail Inventory Intelligence Platform

InventoryAI is an end-to-end retail analytics and inventory intelligence platform that uses machine learning to forecast product demand, identify inventory risk, and recommend replenishment actions.

The goal is to help retailers answer three important questions:

- How much demand should we expect?
- Which products are at risk of running out?
- Which products require immediate replenishment?

---

##  Features

### Demand Forecasting

InventoryAI uses an XGBoost regression model to forecast product-level demand using historical sales, inventory, time-based, and demand-related features.

###  Inventory Risk Detection

Products are classified into three inventory health levels:

- 🔴 **High Risk** — Inventory is below 25% of forecasted weekly demand
- 🟡 **Medium Risk** — Inventory is between 25% and 75% of forecasted weekly demand
- 🟢 **Low Risk** — Inventory is at least 75% of forecasted weekly demand

### 📦 Replenishment Recommendations

InventoryAI converts predictions into actionable business recommendations:

- 🔴 **Urgent Reorder**
- 🟡 **Plan Reorder**
- 🟢 **Healthy Stock**

###  Executive Dashboard

The interactive dashboard provides:

- Current inventory
- Forecasted weekly demand
- High-risk products
- Recommended order quantities
- Average inventory coverage
- Inventory risk distribution
- Store filtering
- Category filtering
- Risk-level filtering
- Product-level inventory decisions

### Dataset Upload

Users can upload compatible retail datasets in:

- CSV
- XLSX
- XLS

format for analysis.

---

##  Machine Learning

### Model

**XGBoost Regressor**

The demand forecasting model uses time-series-aware feature engineering including:

- Lag 1
- Lag 7
- Lag 14
- Lag 28
- 7-day rolling demand
- 28-day rolling demand
- Inventory level
- Price
- Discount
- Calendar features

### Validation Strategy

A chronological train-test split was used instead of a random split to better reflect real-world forecasting.

**Training Period:** 2022

**Testing Period:** 2023

### Model Performance

| Metric | XGBoost |
|---|---:|
| MAE | 69.29 |
| RMSE | 88.84 |
| R² | 0.333 |

The XGBoost model also improved substantially over a simple previous-day demand baseline.

---

## Project Architecture

```text
Retail Dataset
      ↓
Data Validation
      ↓
Feature Engineering
      ↓
Time-Series Features
      ↓
XGBoost Demand Forecasting
      ↓
Inventory Decision Engine
      ↓
Risk Classification
      ↓
Replenishment Recommendation
      ↓
Streamlit Dashboard
```

---

## Business Decision Logic

InventoryAI converts machine-learning predictions into simple business decisions.

```text
Forecasted Weekly Demand
          ↓
Current Inventory
          ↓
Inventory Coverage %
          ↓
Risk Classification
          ↓
Replenishment Recommendation
```

### Decision Rules

| Inventory Coverage | Risk Level | Recommendation |
|---|---|---|
| < 25% | 🔴 High | Urgent Reorder |
| 25% – < 75% | 🟡 Medium | Plan Reorder |
| ≥ 75% | 🟢 Low | Healthy Stock |

### Recommended Order Quantity

```text
Recommended Order Qty
=
Forecasted Weekly Demand - Current Inventory
```

If the calculated quantity is negative, the recommended order quantity is set to zero.

---

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- XGBoost
- Plotly
- Streamlit
- GitHub

---

##  Project Structure

```text
InventoryAI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── InventoryAI_Final_Dataset.csv
│
└── model/
    └── inventoryai_xgb_model.json
```

---

## Business Value

InventoryAI demonstrates how machine learning can be converted into practical retail business decisions.

The platform helps retailers:

- Identify potentially understocked products
- Prioritize replenishment actions
- Forecast upcoming demand
- Monitor inventory health
- Support data-driven purchasing decisions

---

##  Project Limitations

This project is developed as an analytics and decision-support MVP.

The current dataset is synthetic, so the reported model performance should not be interpreted as production performance on real-world retail data.

The inventory thresholds are business assumptions and can be customized according to supplier lead time, service-level targets, and product requirements.

---

## Future Improvements

- Multi-step demand forecasting
- Dynamic safety-stock calculation
- Supplier lead-time integration
- Real-time inventory integration
- Automated purchase-order generation
- Advanced anomaly detection
- Power BI integration
- Authentication and multi-user SaaS functionality

---

## Author

**Pradeep Kalasagond**

Data Science & Analytics | Machine Learning | Business Intelligence

---

## Project Status

**MVP — Ready for Deployment**

InventoryAI is being developed as a real-world retail intelligence product rather than a standalone machine-learning notebook.

- ✅ Demand forecasting
- ✅ Inventory risk detection
- ✅ Replenishment recommendations
- ✅ Interactive Streamlit dashboard
- ✅ Dataset upload
- ✅ Product-level inventory decisions
- ✅ Downloadable inventory report
- ✅ Public cloud deployment
