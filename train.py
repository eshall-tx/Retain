"""
Model Training & Evaluation Script for Retain.
1. Trains exactly 3 models: Logistic Regression, Random Forest, XGBoost.
2. Evaluates with 80/20 stratified split & 5-fold CV: ROC-AUC, accuracy, precision, recall, F1.
3. Selects best model by ROC-AUC and calibrates predicted probabilities.
4. Computes permutation importance and personalizes top 3 risk reasons per customer.
5. Exports predictions to data/predictions.csv and Power BI artifacts to powerbi/.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure root directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data import (
    find_dataset_file,
    load_raw_data,
    clean_data,
    split_data,
    get_feature_columns
)
from src.model import (
    build_pipeline,
    get_models,
    evaluate_pipeline,
    calibrate_model,
    compute_permutation_importance,
    generate_out_of_fold_probabilities,
    save_artifacts
)


def compute_customer_risk_reasons(df_clean: pd.DataFrame, importance_df: pd.DataFrame) -> list:
    """
    Computes the top 3 personalized risk drivers for each customer by combining
    global permutation feature importance weights with the customer's actual feature values.
    """
    # Create importance lookup dict
    imp_map = dict(zip(importance_df["feature"], importance_df["importance_mean"]))
    
    reasons_list = []

    # Column name mappings
    contract_col = "Contract" if "Contract" in df_clean.columns else "contract"
    tenure_col = "Tenure Months" if "Tenure Months" in df_clean.columns else "tenure"
    monthly_col = "Monthly Charges" if "Monthly Charges" in df_clean.columns else "MonthlyCharges"
    net_col = "Internet Service" if "Internet Service" in df_clean.columns else "InternetService"
    support_col = "Tech Support" if "Tech Support" in df_clean.columns else "TechSupport"
    security_col = "Online Security" if "Online Security" in df_clean.columns else "OnlineSecurity"
    payment_col = "Payment Method" if "Payment Method" in df_clean.columns else "PaymentMethod"
    dep_col = "Dependents" if "Dependents" in df_clean.columns else "dependents"

    for _, row in df_clean.iterrows():
        candidate_drivers = []

        # 1. Contract
        c_val = str(row.get(contract_col, ""))
        c_imp = imp_map.get(contract_col, 0.04)
        if "Month" in c_val or "month" in c_val:
            candidate_drivers.append((c_imp * 1.5, f"Month-to-month contract (flexible commitment)"))
        elif "One" in c_val:
            candidate_drivers.append((c_imp * 0.2, f"One-year contract renewal horizon"))

        # 2. Tenure
        t_val = float(row.get(tenure_col, 12))
        t_imp = imp_map.get(tenure_col, 0.03)
        if t_val <= 6:
            candidate_drivers.append((t_imp * 1.4, f"Early account lifecycle ({int(t_val)} months tenure)"))
        elif t_val <= 18:
            candidate_drivers.append((t_imp * 0.8, f"Sub-2-year tenure ({int(t_val)} months)"))

        # 3. Internet Service
        net_val = str(row.get(net_col, ""))
        net_imp = imp_map.get(net_col, 0.02)
        if "Fiber" in net_val:
            candidate_drivers.append((net_imp * 1.2, "Fiber optic service with high price sensitivity"))

        # 4. Monthly Charges
        mc_val = float(row.get(monthly_col, 65))
        mc_imp = imp_map.get(monthly_col, 0.02)
        if mc_val >= 85:
            candidate_drivers.append((mc_imp * 1.3, f"High monthly billing tier (${mc_val:.0f}/mo)"))
        elif mc_val >= 70:
            candidate_drivers.append((mc_imp * 0.8, f"Above-median monthly charges (${mc_val:.0f}/mo)"))

        # 5. Tech Support & Security
        sup_val = str(row.get(support_col, ""))
        sup_imp = imp_map.get(support_col, 0.01)
        if sup_val == "No":
            candidate_drivers.append((sup_imp * 1.1, "No dedicated technical support add-on"))

        sec_val = str(row.get(security_col, ""))
        sec_imp = imp_map.get(security_col, 0.01)
        if sec_val == "No":
            candidate_drivers.append((sec_imp * 1.0, "No online security package"))

        # 6. Payment Method
        pay_val = str(row.get(payment_col, ""))
        pay_imp = imp_map.get(payment_col, 0.01)
        if "Electronic check" in pay_val:
            candidate_drivers.append((pay_imp * 1.0, "Manual electronic check payment method"))

        # 7. Dependents
        dep_val = str(row.get(dep_col, ""))
        dep_imp = imp_map.get(dep_col, 0.015)
        if dep_val == "No":
            candidate_drivers.append((dep_imp * 0.7, "Independent subscriber with no family plan"))

        # Sort candidate drivers by score descending
        candidate_drivers.sort(key=lambda x: x[0], reverse=True)
        top_reasons = [item[1] for item in candidate_drivers[:3]]
        
        # Fallback if customer had very few triggers
        while len(top_reasons) < 3:
            top_reasons.append("Standard account usage profile")

        reasons_list.append("; ".join(top_reasons))

    return reasons_list


def run_training():
    print("=" * 64)
    print(" Retain. | Model Training, Calibration & Power BI Export Pipeline")
    print("=" * 64)

    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    powerbi_dir = os.path.join(base_dir, "powerbi")
    data_path = find_dataset_file(data_dir)
    predictions_path = os.path.join(data_dir, "predictions.csv")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(powerbi_dir, exist_ok=True)

    # 1. Load and clean data
    print(f"\n[1/6] Loading data from {os.path.basename(data_path)}...")
    df_raw = load_raw_data(data_path)
    print(f"      Loaded {len(df_raw)} rows, {len(df_raw.columns)} raw columns.")

    print("[2/6] Cleaning data and removing outcome leakage...")
    df_clean = clean_data(df_raw)
    churn_rate = float(df_clean["Churn"].mean() * 100)
    print(f"      Cleaned dataset churn rate: {churn_rate:.2f}%")

    num_cols, cat_cols, feature_cols = get_feature_columns(df_clean)
    print(f"      Predictive features ({len(feature_cols)} total): {len(num_cols)} numeric, {len(cat_cols)} categorical.")

    # 2. Train/Test split
    print("[3/6] Performing stratified 80/20 train/test split...")
    X_train, X_test, y_train, y_test, _, _ = split_data(df_clean, test_size=0.2, random_state=42)
    print(f"      Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # 3. Train exactly 3 models
    print("\n[4/6] Training and comparing candidate models (Logistic Regression, Random Forest, XGBoost)...")
    models = get_models()
    trained_pipelines = {}
    results = {}

    for name, clf in models.items():
        print(f"      --> Training {name}...")
        pipe = build_pipeline(clf, num_cols, cat_cols)
        pipe.fit(X_train, y_train)
        metrics = evaluate_pipeline(pipe, X_test, y_test)
        trained_pipelines[name] = pipe
        results[name] = metrics
        print(f"          ROC-AUC:  {metrics['roc_auc']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
        print(f"          Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

    # Determine best model by ROC-AUC
    winner_name = max(results, key=lambda k: results[k]["roc_auc"])
    winner_pipe = trained_pipelines[winner_name]
    print(f"\n      * Best model selected by ROC-AUC: {winner_name} (ROC-AUC: {results[winner_name]['roc_auc']:.4f})")

    # 4. Calibrate the best model
    print(f"\n[5/6] Calibrating probabilities for {winner_name} (5-fold Sigmoid Calibration)...")
    calibrated_winner = calibrate_model(
        build_pipeline(get_models()[winner_name], num_cols, cat_cols),
        X_train,
        y_train,
        cv=5
    )
    calibrated_metrics = evaluate_pipeline(calibrated_winner, X_test, y_test)
    print(f"      Calibrated {winner_name} Test ROC-AUC: {calibrated_metrics['roc_auc']:.4f}")

    # Permutation feature importance on test set
    print("      Computing permutation feature importance...")
    importance_df = compute_permutation_importance(winner_pipe, X_test, y_test, n_repeats=10, random_state=42)

    # 5. Out-of-fold honest probabilities for all 7,043 customers
    print("\n[6/6] Generating 5-fold cross-validated out-of-fold probabilities & customer risk reasons...")
    X_all = df_clean[feature_cols]
    y_all = df_clean["Churn"]
    oof_probs = generate_out_of_fold_probabilities(
        build_pipeline(get_models()[winner_name], num_cols, cat_cols),
        X_all,
        y_all,
        cv=5,
        random_state=42
    )

    # Compute personalized top 3 reasons per customer
    top_reasons_per_customer = compute_customer_risk_reasons(df_clean, importance_df)

    # Standardize column naming
    cid_col = "CustomerID" if "CustomerID" in df_clean.columns else "customerID"
    monthly_col = "Monthly Charges" if "Monthly Charges" in df_clean.columns else "MonthlyCharges"
    total_col = "Total Charges" if "Total Charges" in df_clean.columns else "TotalCharges"
    tenure_col = "Tenure Months" if "Tenure Months" in df_clean.columns else "tenure"
    net_col = "Internet Service" if "Internet Service" in df_clean.columns else "InternetService"
    contract_col = "Contract"

    # Risk band categorization
    def assign_risk_band(prob):
        if prob >= 0.50:
            return "High"
        elif prob >= 0.25:
            return "Medium"
        return "Low"

    risk_bands = [assign_risk_band(p) for p in oof_probs]

    # Build predictions dataframe
    predictions_dict = {
        "customerID": df_clean[cid_col].astype(str),
        "MonthlyCharges": pd.to_numeric(df_clean[monthly_col], errors="coerce").fillna(0.0),
        "TotalCharges": pd.to_numeric(df_clean[total_col], errors="coerce").fillna(0.0),
        "tenure": pd.to_numeric(df_clean[tenure_col], errors="coerce").fillna(0).astype(int),
        "Contract": df_clean[contract_col].astype(str),
        "InternetService": df_clean[net_col].astype(str),
        "churn_prob": np.round(oof_probs, 4),
        "risk_band": risk_bands,
        "actual_churn": df_clean["Churn"].astype(int),
        "top_reasons": top_reasons_per_customer
    }

    if "CLTV" in df_clean.columns:
        predictions_dict["CLTV"] = pd.to_numeric(df_clean["CLTV"], errors="coerce").fillna(0.0)

    predictions_df = pd.DataFrame(predictions_dict)
    predictions_df.to_csv(predictions_path, index=False)
    print(f"      Predictions saved to {predictions_path} ({len(predictions_df)} rows).")

    # 6. Export Power BI artifacts
    print(f"\n      Exporting Power BI data files to {powerbi_dir}...")
    
    # 6a. customers_scored.csv
    powerbi_customers = pd.DataFrame({
        "CustomerID": predictions_df["customerID"],
        "Contract": predictions_df["Contract"],
        "TenureMonths": predictions_df["tenure"],
        "MonthlyCharges": predictions_df["MonthlyCharges"],
        "InternetService": predictions_df["InternetService"],
        "ChurnProbability": predictions_df["churn_prob"],
        "RiskBand": predictions_df["risk_band"],
        "ActualChurn": predictions_df["actual_churn"],
        "TopReasons": predictions_df["top_reasons"]
    })
    powerbi_customers.to_csv(os.path.join(powerbi_dir, "customers_scored.csv"), index=False)

    # 6b. model_comparison.csv
    comparison_rows = []
    for m_name, m_stats in results.items():
        comparison_rows.append({
            "Model": m_name,
            "ROC_AUC": m_stats["roc_auc"],
            "Accuracy": m_stats["accuracy"],
            "Precision": m_stats["precision"],
            "Recall": m_stats["recall"],
            "F1_Score": m_stats["f1"],
            "IsBest": 1 if m_name == winner_name else 0
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(powerbi_dir, "model_comparison.csv"), index=False)

    # 6c. feature_importance.csv
    feature_imp_export = importance_df.head(15).copy()
    feature_imp_export.columns = ["Feature", "ImportanceMean", "ImportanceStd"]
    feature_imp_export.to_csv(os.path.join(powerbi_dir, "feature_importance.csv"), index=False)

    # 6d. POWERBI_GUIDE.md
    guide_path = os.path.join(powerbi_dir, "POWERBI_GUIDE.md")
    write_powerbi_guide(guide_path)
    print(f"      Power BI documentation generated at {guide_path}.")

    # 7. Save model and metrics bundle
    model_save_path = os.path.join(models_dir, "model.joblib")
    metrics_save_path = os.path.join(models_dir, "metrics.json")

    # Extract clean dataset insights
    m2m_churn = float(predictions_df[predictions_df["Contract"].str.contains("Month", case=False)]["actual_churn"].mean() * 100)
    fiber_churn = float(predictions_df[predictions_df["InternetService"].str.contains("Fiber", case=False)]["actual_churn"].mean() * 100)
    short_tenure_churn = float(predictions_df[predictions_df["tenure"] <= 12]["actual_churn"].mean() * 100)

    bundle = {
        "winner_model": winner_name,
        "models_evaluated": results,
        "calibrated_winner_metrics": calibrated_metrics,
        "permutation_importance": importance_df.head(10).to_dict(orient="records"),
        "insights": {
            "month_to_month_churn_pct": round(m2m_churn, 1),
            "fiber_optic_churn_pct": round(fiber_churn, 1),
            "short_tenure_churn_pct": round(short_tenure_churn, 1)
        },
        "dataset_summary": {
            "dataset_file": os.path.basename(data_path),
            "total_customers": len(df_clean),
            "churn_rate": round(churn_rate, 2),
            "avg_monthly_charges": round(float(predictions_df["MonthlyCharges"].mean()), 2),
            "total_monthly_revenue": round(float(predictions_df["MonthlyCharges"].sum()), 2),
            "revenue_at_risk": round(float(predictions_df[predictions_df["actual_churn"] == 1]["MonthlyCharges"].sum()), 2),
            "high_risk_count": int((predictions_df["churn_prob"] >= 0.5).sum()),
            "medium_risk_count": int(((predictions_df["churn_prob"] >= 0.25) & (predictions_df["churn_prob"] < 0.5)).sum()),
            "low_risk_count": int((predictions_df["churn_prob"] < 0.25).sum())
        }
    }

    # Save calibrated model pipeline
    save_artifacts(calibrated_winner, bundle, model_save_path, metrics_save_path)
    print(f"      Saved calibrated pipeline to {model_save_path}")
    print(f"      Saved metrics to {metrics_save_path}")

    print("\n" + "=" * 64)
    print(f" Training and Power BI export completed successfully!")
    print(f" Calibrated model: {winner_name} (ROC-AUC {calibrated_metrics['roc_auc']:.4f})")
    print("=" * 64 + "\n")


def write_powerbi_guide(guide_path: str):
    """Writes concise, actionable steps for importing Retain assets into Power BI Desktop."""
    content = """# Power BI Desktop Integration Guide: Retain.

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
"""
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    run_training()
