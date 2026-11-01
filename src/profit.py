"""
Profit optimization module for Retain.
Translates statistical churn probabilities into business revenue and net profit.
"""

import pandas as pd
import numpy as np


def compute_expected_profit(
    df: pd.DataFrame,
    offer_cost: float = 20.0,
    offer_success_rate: float = 0.25,
    months_retained: int = 12,
    cltv_mode: str = "monthly_multiplied"
) -> pd.DataFrame:
    """
    Computes expected profit for each customer if targeted with a retention campaign:
    
    Formula:
        If cltv_mode == 'ibm_cltv' and 'CLTV' in df:
            customer_value = df['CLTV']
        Else:
            customer_value = MonthlyCharges * months_retained

        expected_profit = (churn_prob * offer_success_rate * customer_value) - offer_cost

    Parameters:
        df: DataFrame containing customer columns and 'churn_prob'
        offer_cost: Cost per customer contacted (e.g., $20 voucher/discount)
        offer_success_rate: Probability that customer accepts the offer and stays (e.g., 0.25 = 25%)
        months_retained: Expected duration in months the customer remains after accepting
        cltv_mode: 'monthly_multiplied' (default) or 'ibm_cltv'

    Returns:
        DataFrame sorted by expected_profit in descending order, with cumulative profit.
    """
    ranked = df.copy()

    # Identify Monthly Charges column name flexibly
    monthly_col = "Monthly Charges" if "Monthly Charges" in ranked.columns else "MonthlyCharges"

    # Calculate expected customer lifetime value gained if retained
    if cltv_mode == "ibm_cltv" and "CLTV" in ranked.columns:
        ranked["customer_value"] = pd.to_numeric(ranked["CLTV"], errors="coerce").fillna(0.0)
    else:
        monthly_val = pd.to_numeric(ranked[monthly_col], errors="coerce").fillna(0.0)
        ranked["customer_value"] = monthly_val * months_retained

    # Calculate net expected profit from contacting this specific customer
    ranked["expected_profit"] = (
        ranked["churn_prob"] * offer_success_rate * ranked["customer_value"]
    ) - offer_cost

    # Rank customers by expected profit descending (highest net profit first)
    ranked = ranked.sort_values(by="expected_profit", ascending=False).reset_index(drop=True)
    
    # 1-indexed contact rank
    ranked["contact_rank"] = ranked.index + 1

    # Cumulative calculations for the retention curve
    ranked["cumulative_profit"] = ranked["expected_profit"].cumsum()
    ranked["cumulative_cost"] = ranked["contact_rank"] * offer_cost
    ranked["cumulative_gross_revenue"] = (
        ranked["churn_prob"] * offer_success_rate * ranked["customer_value"]
    ).cumsum()

    return ranked


def find_optimal_cutoff(ranked_df: pd.DataFrame):
    """
    Identifies the optimal number of customers to contact to maximize total expected net profit.
    Finds the peak of the cumulative expected profit curve.
    """
    if ranked_df.empty:
        return {"optimal_count": 0, "max_profit": 0.0, "profitable_customers": 0}

    # Customers where expected profit > 0
    profitable_mask = ranked_df["expected_profit"] > 0
    profitable_count = int(profitable_mask.sum())

    # Peak of cumulative profit curve
    max_idx = ranked_df["cumulative_profit"].idxmax()
    max_profit = float(ranked_df.loc[max_idx, "cumulative_profit"])
    
    # If max profit is negative or zero, optimal is 0 contacts
    if max_profit <= 0:
        optimal_count = 0
        max_profit = 0.0
    else:
        optimal_count = int(ranked_df.loc[max_idx, "contact_rank"])

    return {
        "optimal_count": optimal_count,
        "max_profit": round(max_profit, 2),
        "profitable_customers": profitable_count
    }


def get_campaign_summary(ranked_df: pd.DataFrame, n_contact: int, offer_cost: float):
    """
    Calculates business KPIs for a given number of contacted customers:
    - Expected Net Profit
    - Total Campaign Cost
    - Expected Retained Revenue
    - Return on Investment (ROI %)
    """
    if n_contact <= 0 or ranked_df.empty:
        return {
            "n_contact": 0,
            "expected_net_profit": 0.0,
            "campaign_cost": 0.0,
            "retained_revenue": 0.0,
            "roi_percent": 0.0
        }

    subset = ranked_df.iloc[:n_contact]
    net_profit = float(subset["expected_profit"].sum())
    cost = float(n_contact * offer_cost)
    retained_rev = float((subset["churn_prob"] * subset["customer_value"]).sum())
    roi = (net_profit / cost * 100.0) if cost > 0 else 0.0

    return {
        "n_contact": n_contact,
        "expected_net_profit": round(net_profit, 2),
        "campaign_cost": round(cost, 2),
        "retained_revenue": round(retained_rev, 2),
        "roi_percent": round(roi, 1)
    }
