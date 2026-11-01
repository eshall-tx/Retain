"""
Data processing module for Retain.
Cleans and prepares the IBM Telco Customer Churn dataset (supports both Kaggle Excel
and raw CSV formats). Implements strict data leakage prevention.
"""

import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

# Official IBM sample dataset fallback URL
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# Columns to strictly exclude from model features
# 1. Churn outcome & post-churn leakage
LEAKAGE_COLUMNS = {
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
    "Churn Category",
    "Churn"
}

# 2. Identifiers, constant columns, and noisy geographic features
NOISE_AND_ID_COLUMNS = {
    "CustomerID",
    "customerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude"
}

# 3. Customer Lifetime Value (reserved exclusively for profit calculation, NOT model training)
PROFIT_ONLY_COLUMNS = {
    "CLTV"
}

ALL_EXCLUDED_COLUMNS = LEAKAGE_COLUMNS | NOISE_AND_ID_COLUMNS | PROFIT_ONLY_COLUMNS


def find_dataset_file(data_dir: str) -> str:
    """
    Locates the best available dataset file in data_dir:
    Prefers Telco_customer_churn.xlsx (Kaggle/TanKY version), then Telco-Customer-Churn.csv.
    """
    excel_path = os.path.join(data_dir, "Telco_customer_churn.xlsx")
    if os.path.exists(excel_path):
        return excel_path

    csv_path = os.path.join(data_dir, "Telco-Customer-Churn.csv")
    if os.path.exists(csv_path):
        return csv_path

    return excel_path


def download_dataset_if_missing(filepath: str) -> None:
    """
    Downloads the IBM Telco Customer Churn CSV if no dataset exists locally.
    """
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"Downloading fallback dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, filepath)
        print(f"Dataset saved to {filepath}")


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Reads either .xlsx or .csv from disk into a pandas DataFrame.
    """
    if not os.path.exists(filepath):
        download_dataset_if_missing(filepath)

    if filepath.endswith(".xlsx"):
        return pd.read_excel(filepath)
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans customer data and standardizes target encoding:
    1. Handles blank strings in Total Charges / TotalCharges by coercing to numeric and filling with 0.0.
    2. Converts Churn Label / Churn Value / Churn into standard binary target column 'Churn' (1/0).
    3. Preserves CustomerID, CLTV, and Churn Reason in the dataframe for business use,
       while keeping them separate from model features.
    """
    clean_df = df.copy()

    # Step 1: Standardize Total Charges (coercing whitespace blanks to NaN then 0.0)
    for tc_col in ["Total Charges", "TotalCharges"]:
        if tc_col in clean_df.columns:
            clean_df[tc_col] = pd.to_numeric(
                clean_df[tc_col].astype(str).str.strip(),
                errors="coerce"
            ).fillna(0.0)

    # Step 2: Standardize target column 'Churn' (1 = churned, 0 = stayed)
    if "Churn Value" in clean_df.columns:
        clean_df["Churn"] = pd.to_numeric(clean_df["Churn Value"], errors="coerce").fillna(0).astype(int)
    elif "Churn Label" in clean_df.columns:
        clean_df["Churn"] = (
            clean_df["Churn Label"]
            .astype(str)
            .str.strip()
            .map({"Yes": 1, "No": 0})
            .fillna(0)
            .astype(int)
        )
    elif "Churn" in clean_df.columns:
        if clean_df["Churn"].dtype == object or str(clean_df["Churn"].dtype).startswith("string"):
            clean_df["Churn"] = (
                clean_df["Churn"]
                .astype(str)
                .str.strip()
                .map({"Yes": 1, "No": 0})
                .fillna(0)
                .astype(int)
            )

    return clean_df


def get_feature_columns(df: pd.DataFrame):
    """
    Returns (num_cols, cat_cols, feature_cols) strictly excluding:
    - Target and data leakage columns (Churn Label, Churn Value, Churn Score, Churn Reason, Churn Category)
    - Identifiers and location noise (CustomerID, Count, Country, State, City, Zip Code, Lat Long, Latitude, Longitude)
    - Profit-only metrics (CLTV)
    """
    feature_cols = [col for col in df.columns if col not in ALL_EXCLUDED_COLUMNS]

    # Numeric feature candidates
    num_candidates = [
        "Tenure Months", "Monthly Charges", "Total Charges",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]
    num_cols = [col for col in num_candidates if col in feature_cols]

    # Categorical features are all other remaining features
    cat_cols = [col for col in feature_cols if col not in num_cols]

    return num_cols, cat_cols, feature_cols


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the dataset into stratified training and testing sets.
    Using stratify=y guarantees that the test set has the exact same churn ratio as train.
    """
    num_cols, cat_cols, feature_cols = get_feature_columns(df)
    
    X = df[feature_cols]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, num_cols, cat_cols
