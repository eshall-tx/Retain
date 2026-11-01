# Retain<span style="color:#FF5FA2">.</span> — Customer Churn Prediction & Retention Economics

**Retain.** is an end-to-end data science application combining machine learning (Logistic Regression, Random Forest, XGBoost), probability calibration, and financial retention optimization.

---

## Quickstart

Run the following commands from your terminal:

```bash
cd C:\Users\Lenovo\.gemini\antigravity\scratch\retain
python train.py
streamlit run app.py
```

The web dashboard will open in your browser at `http://localhost:8501`.

---

## Application Structure & Pages

1. **Home (`/`)**:
   - **Hero & Cohort Animation**: Status indicator with calibrated model status, headline, real cohort churn rate, and an animated CSS customer cohort visualization displaying 100 customer dots drifting across an attrition threshold.
   - **Open KPI Strip**: Key numbers (`customers · n = 7,043`, `churn rate`, `monthly revenue at risk`) with custom SVG sparklines.
   - **Interactive Playground (`02 / playground`)**: Test customer profiles live across contract type, tenure, and monthly charges against the calibrated pipeline, featuring an animated semicircle probability gauge and top 3 risk factors.
   - **Data-Driven Insights**: 3 insight entries with colored top rules generated dynamically from the underlying dataset.

2. **Risk Explorer (`/explorer`)**:
   - Filterable customer diagnostics by contract type, internet infrastructure, risk band, and tenure band.
   - Clean, open account table with inline probability bars.
   - Customer inspection panel displaying individual top 3 risk drivers and rule-based recommended retention interventions.

3. **Who to Contact (`/contact`)**:
   - Business economics engine calculating expected net profit:
     $$\text{expected\_profit} = \left(\text{churn\_prob} \times \text{offer\_success\_rate} \times \text{MonthlyCharges} \times \text{months\_retained}\right) - \text{offer\_cost}$$
   - Interactive cumulative expected profit curve annotating the optimal cutoff point.
   - Scored target list with CSV export.

4. **Model Lab (`/lab`)**:
   - Benchmark comparison table comparing **Logistic Regression**, **Random Forest**, and **XGBoost** across ROC-AUC, Accuracy, Precision, Recall, and F1.
   - Multi-model ROC comparison curves.
   - Holdout confusion matrix at 0.50 decision threshold.
   - 10-bin probability calibration curve verifying honest probabilities.
   - Permutation feature importance isolating top churn drivers.

---

## Power BI Integration

The training pipeline automatically exports scored data to `powerbi/`:
- `customers_scored.csv`: Complete scored account table with `ChurnProbability`, `RiskBand`, and `TopReasons`.
- `model_comparison.csv`: Model evaluation metrics.
- `feature_importance.csv`: Ranked permutation importance weights.
- `POWERBI_GUIDE.md`: Step-by-step instructions to load assets into Power BI Desktop and construct executive KPI cards and charts.

---

## Data Leakage Governance & Preprocessing

- **Dataset**: Kaggle IBM Telco Customer Churn dataset (`Telco_customer_churn.xlsx`, 7,043 rows, 33 attributes).
- **Leakage Columns Removed**: `Churn Label`, `Churn Value`, `Churn Score`, `Churn Reason`, and `Churn Category` are strictly excluded from predictive features.
- **Location & Identifier Filtering**: `CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, and `Longitude` are removed to prevent spatial noise and memorization.
- **CLTV Policy**: Customer Lifetime Value (`CLTV`) is excluded from model features to prevent post-hoc bias.
