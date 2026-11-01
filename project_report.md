# Academic Project Report: Retain<span style="color:#ff4d94">.</span>

**Course:** IBM Data Science Professional Assignment  
**Project Title:** Retain. — Customer Churn Prediction and Retention-Profit Optimization  
**Student Submission:** 10-Mark Capstone Implementation  

---

## 1. Problem Definition
Customer attrition (churn) is a major revenue drain for telecommunication service providers. While standard predictive modeling evaluates whether a customer will churn based on arbitrary probability cutoffs (such as 0.5), real-world retention marketing is an economic decision:
1. Contacting every churn-prone customer is wasteful and expensive.
2. Saving high-value accounts yields vastly higher financial impact than saving low-value accounts.
3. Churn interventions carry direct costs (discounts, outreach incentives).

**Objective:** Develop an end-to-end data science application that predicts customer churn probabilities without artificial class weighting, validates probability calibration, and optimizes retention campaigns based on **expected net profit**.

---

## 2. Dataset Overview & Data Cleaning
The application utilizes the **Telco Customer Churn: IBM Dataset** (Kaggle curation by TanKY from IBM Cognos Analytics base samples):
- **Total Records:** 7,043 customer accounts.
- **Total Columns:** 33 raw attributes spanning demographics, geographic details, subscribed services, account financials, and attrition records.
- **Baseline Churn Rate:** 26.54% (1,869 churned, 5,174 retained).
- **Missing Values:** Only `Churn Reason` contained missing values (5,174 nulls, corresponding exclusively to active, non-churned subscribers).

### Data Preprocessing & Leakage Governance (`src/data.py`)
1. **Total Charges Whitespace Anomaly:** The raw `Total Charges` column contained 11 records with whitespace strings (`" "`). Inspection revealed these were new customers with `Tenure Months = 0` who had not yet received their first bill. These values were coerced to numeric and filled with `0.0`.
2. **Binary Target Standardization:** The target variable `Churn Label` (`"Yes"` / `"No"`) and `Churn Value` (`1` / `0`) were mapped into a clean binary integer target `Churn`.
3. **Data Leakage Elimination:** Columns that reveal or derive from the outcome were strictly removed from model features:
   - `Churn Label` & `Churn Value`: Direct target indicators.
   - `Churn Score`: Post-hoc derived attrition risk score.
   - `Churn Reason` & `Churn Category`: Post-mortem exit survey data collected *only after* customer cancellation.
4. **Noisy Location & Identifier Removal:**
   - `CustomerID`: Arbitrary key; excluded to prevent spurious memorization.
   - `Count`, `Country`, `State`: Zero-variance constants (`1`, `"United States"`, `"California"`).
   - `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`: High-cardinality spatial noise causing overfit.
5. **CLTV Policy:** Customer Lifetime Value (`CLTV`) is excluded from ML training features and reserved strictly for customer economic valuation on the profit optimization dashboard.
6. **Stratified Split:** An 80/20 train/test split with stratification on `Churn` yielded **5,634 training samples** and **1,409 test samples** with identical 26.54% churn distribution across 19 clean features (3 numeric, 16 categorical).

---

## 3. Methodology & Modeling
To adhere strictly to production standards and guarantee that raw column names are preserved, all preprocessing and modeling steps were encapsulated in a scikit-learn `Pipeline` paired with a `ColumnTransformer`:
- **Numeric Pipeline:** `StandardScaler` applied to `['Tenure Months', 'Monthly Charges', 'Total Charges']`.
- **Categorical Pipeline:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to all 16 categorical variables.

### Honest Probability Constraint
Unlike standard classification assignments that set `class_weight="balanced"`, this project purposefully leaves `class_weight=None`. While balanced weighting artificially elevates recall, it breaks probability calibration by warping posterior probabilities:
$$P(\text{churn} \mid X)$$
Because net profit calculations rely on the true statistical expectation of revenue, honest probabilities are essential.

### Out-of-Fold Cross-Validation
To generate rankings for the entire customer base without risk of training leakage, 5-fold stratified cross-validation (`cross_val_predict(..., method='predict_proba')`) was used to compute out-of-fold probabilities for all 7,043 customers.

---

## 4. Experimental Results & Diagnostics

### Model Evaluation on Holdout Test Set (1,409 customers)

| Metric | Logistic Regression (Baseline) | Random Forest Classifier (Selected) |
| :--- | :---: | :---: |
| **ROC-AUC** | 0.8489 | **0.8533** |
| **Precision (0.5 threshold)** | 0.6435 | **0.6735** |
| **Recall (0.5 threshold)** | **0.5695** | 0.5241 |
| **Confusion Matrix (0.5)** | `[[917, 118], [161, 213]]` | `[[940, 95], [178, 196]]` |

*Evaluation Analysis:*
- Random Forest achieved the highest ROC-AUC (**0.8533**), demonstrating strong discrimination ability across all classification thresholds.
- At the standard 0.5 threshold, Random Forest showed higher precision (**67.35%** vs 64.35%), generating fewer false alarms (95 vs 118 false positives).

### Probability Calibration
Probability reliability was measured using an empirical 10-bin calibration curve. Both candidate pipelines aligned closely with the theoretical diagonal ($y = x$), confirming that a predicted probability of 0.60 corresponds accurately to an observed ~60% empirical churn frequency.

### Top Churn Drivers (Permutation Feature Importance)
Permutation importance was evaluated on the test set by measuring the drop in ROC-AUC when each feature column is randomly shuffled:
1. **Contract Type** (ROC-AUC drop of **+0.0411**): Month-to-month contracts are overwhelmingly the strongest indicator of churn risk.
2. **Dependents** (ROC-AUC drop of **+0.0253**): Customers without dependents churn at significantly higher rates.
3. **Tenure Months** (ROC-AUC drop of **+0.0219**): Newer subscribers exhibit the highest vulnerability.
4. **Internet Service** (ROC-AUC drop of **+0.0119**): Fiber optic subscribers show higher churn than DSL subscribers.
5. **Total Charges** (ROC-AUC drop of **+0.0117**): Cumulative billing history reflects account maturity.

---

## 5. Retention Profit Optimization Logic (`src/profit.py`)

Rather than intervening on all predicted churners, the system optimizes campaign profitability using the formula:
$$\text{customer\_value} = \text{MonthlyCharges} \times \text{months\_retained}$$
$$\text{expected\_profit} = \left(\text{churn\_prob} \times \text{offer\_success\_rate} \times \text{customer\_value}\right) - \text{offer\_cost}$$

### Business Campaign Simulation (Default Assumptions: \$20 offer, 25% acceptance, 12 months retention)
- **Optimal Contact Volume:** **4,453 customers**
- **Maximum Expected Net Profit:** **\$308,967.84**
- **Total Campaign Spend:** **\$89,060.00**
- **Retained Customer Gross Revenue:** **\$398,027.84**
- **Campaign ROI:** **346.9%**

Beyond 4,453 contacts, customer churn probability or monthly bill amounts are insufficient to cover the \$20 outreach cost, causing cumulative profit to decline.

---

## 6. Project Limitations

1. **Static Sample Dataset:** The IBM Telco dataset is a static educational sample snapshot. In an enterprise setting, streaming telemetry logs (e.g., dropped calls, bandwidth throttling, mobile app logins) provide richer temporal signals.
2. **Sensitivity to Business Assumptions:** The retention profit formula depends on parameters (`offer_cost`, `offer_success_rate`, `months_retained`) that are managerial estimates rather than immutable facts. Users must treat projections as scenario simulations.
3. **Probability Calibration in Drift Conditions:** While the calibration curve validated honest probabilities on the test split, in production environments customer distributions shift over time (covariate drift), requiring regular recalibration.

---

## 7. Extensions & Completed Modules
- **Message Triage (Phase 2 - Completed):** Integrated multi-signal support message classification in `src/triage.py` using TypeSafe AI's Jev model with automatic fallback to local VADER sentiment and lexical keyword rules to tag urgency, refund risk, and human escalation need.
- **Personalized Retention Copy (Phase 3 - Completed):** Built `src/messages.py` supporting IBM Granite 4.0 Nano (~350M parameters, CPU-native) and dynamic contract-aware fallback copy to draft concise retention messages (<60 words) strictly tied to business offers without hallucinating unauthorized discounts.
- **Uplift Modeling (Future Direction):** Moving from conditional churn estimation $P(\text{churn})$ to heterogeneous treatment effect estimation $\tau_i$, predicting which customers *only* stay when given an offer.
