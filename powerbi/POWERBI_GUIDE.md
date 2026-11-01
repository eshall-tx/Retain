# Power BI Desktop Integration Guide: Retain.

This guide provides concise steps to import Retain's scored customer data and model analytics into **Power BI Desktop** and build an executive retention dashboard.

---

## 1. Exported Data Files

All exported tables are located in the `powerbi/` folder:
- **`customers_scored.csv`**: Customer-level table (7,043 rows) containing `CustomerID`, `Contract`, `TenureMonths`, `MonthlyCharges`, `InternetService`, `ChurnProbability`, `RiskBand` (Low / Medium / High), `ActualChurn`, and `TopReasons`.
- **`model_comparison.csv`**: Benchmark metrics across Logistic Regression, Random Forest, and XGBoost (`ROC_AUC`, `Accuracy`, `Precision`, `Recall`, `F1_Score`, `IsBest`).
- **`feature_importance.csv`**: Top predictive churn drivers ranked by permutation importance drop.

---

## 2. Importing Data into Power BI Desktop

1. Open **Power BI Desktop**.
2. Click **Get Data** → **Text/CSV**.
3. Select `powerbi/customers_scored.csv` and click **Load**.
4. Repeat for `powerbi/model_comparison.csv` and `powerbi/feature_importance.csv`.
5. In Model View, confirm `customers_scored` is your primary fact table.

---

## 3. Recommended Visualizations

### Visual 1: Executive KPI Cards
- **Card 1 (Total Customers)**: Field `CustomerID` (Count).
- **Card 2 (Average Churn Probability)**: Field `ChurnProbability` (Average, formatted as percentage).
- **Card 3 (High Risk Customers)**: Measure `CALCULATE(COUNTROWS(customers_scored), customers_scored[RiskBand] = "High")`.
- **Card 4 (Monthly Revenue at Risk)**: Measure `CALCULATE(SUM(customers_scored[MonthlyCharges]), customers_scored[ChurnProbability] >= 0.5)`.

### Visual 2: Churn Probability by Contract Type
- **Visual**: Clustered Column Chart or Bar Chart.
- **X-axis**: `Contract`.
- **Y-axis**: Average of `ChurnProbability`.
- **Data Color**: Primary Pink (`#FF5FA2`).

### Visual 3: Risk Distribution by Tenure Band
- **Calculated Column** in `customers_scored`:
  ```dax
  TenureBand = SWITCH(
      TRUE(),
      customers_scored[TenureMonths] <= 12, "0-12 Mos",
      customers_scored[TenureMonths] <= 24, "13-24 Mos",
      customers_scored[TenureMonths] <= 48, "25-48 Mos",
      "49+ Mos"
  )
  ```
- **Visual**: 100% Stacked Bar Chart.
- **Y-axis**: `TenureBand`.
- **X-axis**: Count of `CustomerID`.
- **Legend**: `RiskBand` (High = `#FF5FA2`, Medium = `#FBBF24`, Low = `#2DD4BF`).

### Visual 4: Priority Action Table (High-Risk Customers)
- **Visual**: Table visual.
- **Columns**: `CustomerID`, `Contract`, `TenureMonths`, `MonthlyCharges`, `ChurnProbability`, `TopReasons`.
- **Visual-level filter**: `RiskBand` is `High` (or `ChurnProbability >= 0.5`).
- **Sort**: `ChurnProbability` descending.

### Visual 5: Model Comparison Chart
- **Visual**: Clustered Bar Chart.
- **Source**: `model_comparison.csv`.
- **Y-axis**: `Model`.
- **X-axis**: `ROC_AUC` and `F1_Score`.

---

## 4. Theme & Palette Settings

For visual consistency with Retain:
- Background: `#0D0913`
- Visual card background: `#161020`
- Primary Accent: `#FF5FA2` (Pink)
- Secondary Accent: `#2DD4BF` (Teal)
- Warning / Mid-risk: `#FBBF24` (Amber)
- Font: **Inter** or **Segoe UI**
