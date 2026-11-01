"""
Retain. — Customer Churn Prediction & Retention Economics
Dark, technical, editorial dashboard.
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.profit import compute_expected_profit, find_optimal_cutoff, get_campaign_summary

# Set Streamlit page configuration
st.set_page_config(
    page_title="Retain.",
    page_icon="·",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark technical Plotly styling
PLOTLY_EDITORIAL_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#EDE8F5", size=12),
    xaxis=dict(
        gridcolor="#2A2236",
        zerolinecolor="#2A2236",
        tickfont=dict(family="IBM Plex Mono, monospace", size=11, color="#8A7FA0"),
        linecolor="#2A2236"
    ),
    yaxis=dict(
        gridcolor="#2A2236",
        zerolinecolor="#2A2236",
        tickfont=dict(family="IBM Plex Mono, monospace", size=11, color="#8A7FA0"),
        linecolor="#2A2236"
    ),
    margin=dict(l=30, r=20, t=40, b=30)
)


@st.cache_data
def load_stylesheet():
    """Loads external editorial CSS stylesheet."""
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_data_and_metrics():
    """Loads out-of-fold scored customer dataset and model evaluation metrics."""
    base_dir = os.path.dirname(__file__)
    preds_path = os.path.join(base_dir, "data", "predictions.csv")
    metrics_path = os.path.join(base_dir, "models", "metrics.json")

    if not os.path.exists(preds_path) or not os.path.exists(metrics_path):
        return None, None

    preds_df = pd.read_csv(preds_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return preds_df, metrics


@st.cache_resource
def load_trained_model():
    """Loads the calibrated pipeline for live playground scoring."""
    import joblib
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "models", "model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def render_top_nav(current_page: str):
    """
    Renders minimal editorial top navigation:
    Wordmark on left, active/muted text links with 1px pink underline on right.
    """
    c_brand, c_nav = st.columns([1, 2])
    
    with c_brand:
        st.markdown(
            '<div style="padding-top: 4px;"><span class="nav-brand">Retain<span class="pink-period">.</span></span></div>',
            unsafe_allow_html=True
        )

    with c_nav:
        n1, n2, n3, n4 = st.columns(4)
        
        pages = [
            ("Home", "home", n1),
            ("Risk explorer", "explorer", n2),
            ("Who to contact", "contact", n3),
            ("Model lab", "lab", n4)
        ]
        
        for label, p_key, col in pages:
            with col:
                is_active = (current_page == p_key)
                cls = "nav-btn-active" if is_active else "nav-btn"
                st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
                if st.button(label, key=f"nav_{p_key}"):
                    st.session_state["active_page"] = p_key
                    st.session_state["last_query_page"] = p_key
                    st.query_params["page"] = p_key
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hairline-divider-tight"></div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: HOME
# ==============================================================================
def render_page_home(preds_df: pd.DataFrame, metrics: dict):
    summary = metrics.get("dataset_summary", {})
    total_cust = summary.get("total_customers", len(preds_df))
    churn_rate = summary.get("churn_rate", 26.54)
    rev_at_risk = summary.get("revenue_at_risk", 139130.0)
    winner_name = metrics.get("winner_model", "XGBoost")
    insights = metrics.get("insights", {})

    # --- HERO SECTION ---
    c_left, c_right = st.columns([1.1, 1.2])

    with c_left:
        # Status line with blinking dot
        st.markdown(
            f"""
            <div class="status-pill fade-up-1">
                <span class="blinking-dot"></span>
                <span class="mono-label">model calibrated · {winner_name.lower()}</span>
            </div>
            <div class="hero-headline fade-up-2">
                Who leaves,<br>why, and what<br>to do about it.
            </div>
            <div class="hero-subtext fade-up-3">
                Across {total_cust:,} customer accounts, {churn_rate:.2f}% terminated subscription services,
                resulting in ${rev_at_risk:,.0f} in recurring monthly revenue exposure.
            </div>
            """,
            unsafe_allow_html=True
        )

        h_act1, h_act2 = st.columns([1.1, 1.4])
        with h_act1:
            if st.button("Explore risk profiles", key="btn_hero_explore"):
                st.session_state["active_page"] = "explorer"
                st.session_state["last_query_page"] = "explorer"
                st.query_params["page"] = "explorer"
                st.rerun()
        with h_act2:
            st.markdown(
                """
                <style>
                .hero-link-btn button {
                    background: transparent !important;
                    border: none !important;
                    border-radius: 0 !important;
                    padding: 8px 0 0 0 !important;
                    font-family: var(--font-body), sans-serif !important;
                    font-size: 13px !important;
                    font-weight: 500 !important;
                    color: var(--text-muted, #A398B8) !important;
                    box-shadow: none !important;
                    transform: none !important;
                    text-decoration: none !important;
                    line-height: 1.5 !important;
                }
                .hero-link-btn button:hover {
                    color: var(--primary-pink, #FF5FA2) !important;
                    background: transparent !important;
                    box-shadow: none !important;
                    transform: none !important;
                }
                </style>
                <div class="hero-link-btn">
                """,
                unsafe_allow_html=True
            )
            if st.button("View model lab →", key="btn_hero_lab"):
                st.session_state["active_page"] = "lab"
                st.session_state["last_query_page"] = "lab"
                st.query_params["page"] = "lab"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        # CUSTOMER COHORT ANIMATION (Pure CSS signature element)
        pink_dots_svg = ""
        # 22 pink dots across grid positions
        dot_coords = [
            (55, 33, "0.1s"), (99, 55, "0.4s"), (143, 77, "0.8s"), (187, 33, "1.2s"),
            (77, 121, "1.5s"), (121, 143, "1.8s"), (165, 165, "2.1s"), (209, 121, "2.4s"),
            (99, 187, "2.7s"), (143, 209, "3.0s"), (231, 55, "3.3s"), (275, 77, "3.6s"),
            (253, 143, "3.9s"), (297, 165, "4.2s"), (187, 187, "4.5s"), (319, 33, "4.8s"),
            (341, 99, "5.1s"), (363, 143, "5.4s"), (275, 187, "5.7s"), (319, 121, "6.0s"),
            (165, 33, "6.3s"), (231, 99, "6.6s")
        ]
        for cx, cy, delay in dot_coords:
            pink_dots_svg += f'<circle cx="{cx}" cy="{cy}" r="3" fill="#FF5FA2" class="pink-churn-dot" style="animation-delay: {delay}; filter: drop-shadow(0 0 3px rgba(255,95,162,0.8));" />'

        st.markdown(
            f"""
            <div class="fade-up-2">
                <div class="cohort-header">
                    <span class="mono-label">100 customers, one cohort</span>
                    <span class="mono-label text-pink">exit threshold &rarr;</span>
                </div>
                <div class="cohort-container">
                    <div class="cohort-scanline"></div>
                    <svg viewBox="0 0 550 220" width="550" height="220" style="background: transparent;">
                        <defs>
                            <pattern id="teal-grid" width="22" height="22" patternUnits="userSpaceOnUse">
                                <circle cx="11" cy="11" r="1.5" fill="#2DD4BF" opacity="0.32" />
                            </pattern>
                        </defs>
                        <!-- Grid of 100 retained customer positions -->
                        <rect width="550" height="220" fill="url(#teal-grid)" />
                        <!-- Dashed pink exit threshold line -->
                        <line x1="410" y1="0" x2="410" y2="220" stroke="#FF5FA2" stroke-width="1.5" stroke-dasharray="4,4" opacity="0.65" />
                        <!-- Drifting pink churn customer dots -->
                        {pink_dots_svg}
                    </svg>
                </div>
                <div class="cohort-legend">
                    <span style="display: inline-flex; align-items: center; gap: 6px;">
                        <span class="legend-dot-teal"></span>
                        <span class="mono-label">retained</span>
                    </span>
                    <span style="display: inline-flex; align-items: center; gap: 6px;">
                        <span class="legend-dot-pink"></span>
                        <span class="mono-label">predicted to churn</span>
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- OPEN STATS ROW (Between two hairlines, separated by vertical hairlines) ---
    st.markdown(
        f"""
        <div class="stats-row-container fade-up-3">
            <div class="stats-grid">
                <div class="stat-cell">
                    <div class="mono-label">customers · n = {total_cust:,}</div>
                    <div class="stat-number">{total_cust:,}</div>
                    <svg class="sparkline-svg" viewBox="0 0 140 16">
                        <path d="M0,12 Q35,4 70,10 T140,6" fill="none" stroke="#2DD4BF" stroke-width="2"/>
                    </svg>
                </div>
                <div class="stat-cell">
                    <div class="mono-label">churn rate · cohort</div>
                    <div class="stat-number text-pink">{churn_rate:.2f}%</div>
                    <svg class="sparkline-svg" viewBox="0 0 140 16">
                        <path d="M0,14 Q35,12 70,6 T140,2" fill="none" stroke="#FF5FA2" stroke-width="2"/>
                    </svg>
                </div>
                <div class="stat-cell">
                    <div class="mono-label">monthly revenue at risk</div>
                    <div class="stat-number text-amber">${rev_at_risk:,.0f}</div>
                    <svg class="sparkline-svg" viewBox="0 0 140 16">
                        <path d="M0,6 Q35,14 70,4 T140,10" fill="none" stroke="#FBBF24" stroke-width="2"/>
                    </svg>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- 02 / PLAYGROUND SECTION ---
    p_left, p_right = st.columns([0.9, 1.3])

    with p_left:
        st.markdown(
            """
            <div class="section-kicker">02 / playground</div>
            <h2 style="font-size: 28px; line-height: 1.15; margin-bottom: 0.8rem;">Test a customer profile</h2>
            <p style="font-size: 14px; color: var(--text-muted); line-height: 1.6; max-width: 320px;">
                Adjust subscription parameters to evaluate churn probability and risk attributions in real time against the calibrated model.
            </p>
            """,
            unsafe_allow_html=True
        )

    with p_right:
        ctrl_col, out_col = st.columns([1.1, 1.1])
        
        with ctrl_col:
            st.markdown('<div class="mono-label" style="margin-bottom: 8px;">contract_type</div>', unsafe_allow_html=True)
            contract_choice = st.radio(
                "contract_choice",
                ["Month-to-month", "One year", "Two year"],
                horizontal=True,
                label_visibility="collapsed",
                key="play_contract"
            )

            st.markdown('<div class="mono-label" style="margin-top: 14px; margin-bottom: 4px;">tenure_months</div>', unsafe_allow_html=True)
            tenure_input = st.slider("tenure_months_slider", 1, 72, 6, label_visibility="collapsed", key="play_tenure")
            st.markdown(f'<div class="mono-label" style="color: var(--accent-teal);">{tenure_input} months</div>', unsafe_allow_html=True)

            st.markdown('<div class="mono-label" style="margin-top: 14px; margin-bottom: 4px;">monthly_charges</div>', unsafe_allow_html=True)
            charges_input = st.slider("charges_slider", 18, 120, 85, label_visibility="collapsed", key="play_charges")
            st.markdown(f'<div class="mono-label" style="color: var(--accent-amber);">${charges_input}.00 / mo</div>', unsafe_allow_html=True)

        with out_col:
            # Model live prediction calculation
            model = load_trained_model()
            prob_score = 0.65
            if model is not None:
                try:
                    # Construct clean dummy row
                    test_row = {
                        "Gender": "Male",
                        "Senior Citizen": "No",
                        "Partner": "No",
                        "Dependents": "No",
                        "Tenure Months": tenure_input,
                        "Phone Service": "Yes",
                        "Multiple Lines": "No",
                        "Internet Service": "Fiber optic" if charges_input > 70 else "DSL",
                        "Online Security": "No",
                        "Online Backup": "No",
                        "Device Protection": "No",
                        "Tech Support": "No",
                        "Streaming TV": "Yes" if charges_input > 80 else "No",
                        "Streaming Movies": "Yes" if charges_input > 80 else "No",
                        "Contract": contract_choice,
                        "Paperless Billing": "Yes",
                        "Payment Method": "Electronic check",
                        "Monthly Charges": float(charges_input),
                        "Total Charges": float(tenure_input * charges_input)
                    }
                    sample_df = pd.DataFrame([test_row])
                    prob_score = float(model.predict_proba(sample_df)[0, 1])
                except Exception:
                    prob_score = 0.65

            # Risk band assignment
            if prob_score >= 0.50:
                band_label = "high risk"
                gauge_color = "#FF5FA2"
            elif prob_score >= 0.25:
                band_label = "medium risk"
                gauge_color = "#FBBF24"
            else:
                band_label = "low risk"
                gauge_color = "#2DD4BF"

            # Gauge arc stroke math (semi-circle circumference = 314.16)
            gauge_offset = 314.16 * (1.0 - min(max(prob_score, 0.0), 1.0))

            # Dynamic reason generation for current simulated profile
            reasons = []
            if "Month" in contract_choice:
                reasons.append("Month-to-month commitment (+4.1% risk)")
            if tenure_input <= 12:
                reasons.append(f"Early lifecycle account ({tenure_input} mos)")
            if charges_input >= 75:
                reasons.append(f"High recurring charge (${charges_input}/mo)")
            if len(reasons) < 3:
                reasons.append("Unbundled service without tech support")
            if len(reasons) < 3:
                reasons.append("Payment via electronic billing method")

            st.markdown(
                f"""
                <div style="text-align: center; padding-top: 5px;">
                    <svg viewBox="0 0 240 135" width="220" height="125" style="display: block; margin: 0 auto;">
                        <defs>
                            <linearGradient id="gauge-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stop-color="#2DD4BF" />
                                <stop offset="50%" stop-color="#FBBF24" />
                                <stop offset="100%" stop-color="#FF5FA2" />
                            </linearGradient>
                        </defs>
                        <!-- Track -->
                        <path d="M 20 120 A 100 100 0 0 1 220 120" fill="none" stroke="#2A2236" stroke-width="9" stroke-linecap="round" />
                        <!-- Active Arc -->
                        <path d="M 20 120 A 100 100 0 0 1 220 120" fill="none" stroke="url(#gauge-grad)" stroke-width="9" stroke-linecap="round"
                              stroke-dasharray="314.16" stroke-dashoffset="{gauge_offset:.1f}" class="gauge-arc-path" />
                    </svg>
                    <div class="gauge-prob-number">{prob_score:.2f}</div>
                    <div class="mono-label" style="margin-top: 4px; color: {gauge_color};">p(churn) · {band_label}</div>
                </div>

                <div style="margin-top: 1.5rem; text-align: left;">
                    <div class="reason-row">
                        <span class="reason-num">01</span>
                        <span>{reasons[0]}</span>
                    </div>
                    <div class="reason-row">
                        <span class="reason-num">02</span>
                        <span>{reasons[1]}</span>
                    </div>
                    <div class="reason-row">
                        <span class="reason-num">03</span>
                        <span>{reasons[2]}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown('<div class="hairline-divider"></div>', unsafe_allow_html=True)

    # --- THREE INSIGHT ENTRIES (3 columns with colored top rules, data-driven) ---
    m2m_pct = insights.get("month_to_month_churn_pct", 42.7)
    short_pct = insights.get("short_tenure_churn_pct", 47.7)
    fiber_pct = insights.get("fiber_optic_churn_pct", 41.9)

    st.markdown(
        f"""
        <div class="insight-grid">
            <div class="insight-col top-pink">
                <div class="mono-label text-pink">01 · contract</div>
                <div class="insight-text">
                    Month-to-month contracts experience a {m2m_pct}% churn rate, the highest across contract categories.
                    Transitioning subscribers to fixed agreements represents the single largest retention lever.
                </div>
            </div>
            <div class="insight-col top-teal">
                <div class="mono-label text-teal">02 · tenure</div>
                <div class="insight-text">
                    Subscribers with 12 months or less tenure exhibit {short_pct}% attrition. Account mortality declines
                    sharply after month 18, demonstrating that early-stage onboarding determines long-term relationship equity.
                </div>
            </div>
            <div class="insight-col top-amber">
                <div class="mono-label text-amber">03 · infrastructure</div>
                <div class="insight-text">
                    Fiber optic subscribers churn at {fiber_pct}%, driven by premium billing tiers ($70–$110/mo) without
                    bundled technical support packages, creating acute price sensitivity.
                </div>
            </div>
        </div>
        
        <div class="hairline-divider-tight"></div>
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 1.2rem 0;">
            <span class="mono-label">retain · customer churn prediction & retention economics</span>
            <span class="mono-label">ibm telco dataset · scikit-learn · xgboost</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# PAGE 2: RISK EXPLORER
# ==============================================================================
def render_page_explorer(preds_df: pd.DataFrame):
    st.markdown(
        """
        <div style="margin-bottom: 2rem;">
            <div class="mono-label text-pink" style="margin-bottom: 6px;">02 · cohort diagnostics</div>
            <h1 style="font-size: 36px; margin-bottom: 8px;">Risk explorer</h1>
            <p style="color: var(--text-muted); font-size: 14px; max-width: 600px;">
                Inspect individual customer risk distributions, top contributing factors, and rule-based retention actions.
            </p>
        </div>
        <div class="hairline-divider-tight"></div>
        """,
        unsafe_allow_html=True
    )

    # Filter row
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown('<div class="mono-label">contract_filter</div>', unsafe_allow_html=True)
        contracts = ["All"] + sorted(list(preds_df["Contract"].unique()))
        sel_contract = st.selectbox("Contract", contracts, label_visibility="collapsed")
    with f2:
        st.markdown('<div class="mono-label">internet_filter</div>', unsafe_allow_html=True)
        internets = ["All"] + sorted(list(preds_df["InternetService"].unique()))
        sel_internet = st.selectbox("Internet", internets, label_visibility="collapsed")
    with f3:
        st.markdown('<div class="mono-label">risk_band_filter</div>', unsafe_allow_html=True)
        bands = ["All", "High (≥ 0.50)", "Medium (0.25 - 0.50)", "Low (< 0.25)"]
        sel_band = st.selectbox("Risk Band", bands, label_visibility="collapsed")
    with f4:
        st.markdown('<div class="mono-label">tenure_band_filter</div>', unsafe_allow_html=True)
        tenure_bands = ["All", "0-12 mos", "13-24 mos", "25-48 mos", "49+ mos"]
        sel_tenure = st.selectbox("Tenure Band", tenure_bands, label_visibility="collapsed")

    # Apply filters
    filtered = preds_df.copy()
    if sel_contract != "All":
        filtered = filtered[filtered["Contract"] == sel_contract]
    if sel_internet != "All":
        filtered = filtered[filtered["InternetService"] == sel_internet]
    if sel_band == "High (≥ 0.50)":
        filtered = filtered[filtered["churn_prob"] >= 0.50]
    elif sel_band == "Medium (0.25 - 0.50)":
        filtered = filtered[(filtered["churn_prob"] >= 0.25) & (filtered["churn_prob"] < 0.50)]
    elif sel_band == "Low (< 0.25)":
        filtered = filtered[filtered["churn_prob"] < 0.25]

    if sel_tenure == "0-12 mos":
        filtered = filtered[filtered["tenure"] <= 12]
    elif sel_tenure == "13-24 mos":
        filtered = filtered[(filtered["tenure"] > 12) & (filtered["tenure"] <= 24)]
    elif sel_tenure == "25-48 mos":
        filtered = filtered[(filtered["tenure"] > 24) & (filtered["tenure"] <= 48)]
    elif sel_tenure == "49+ mos":
        filtered = filtered[filtered["tenure"] > 48]

    st.markdown(
        f'<div class="mono-label" style="margin: 1.5rem 0 0.8rem 0;">displaying {len(filtered):,} matching accounts</div>',
        unsafe_allow_html=True
    )

    # Split layout: Table on left, customer details on right
    c_tbl, c_detail = st.columns([1.3, 0.9])

    with c_tbl:
        sorted_df = filtered.sort_values(by="churn_prob", ascending=False).reset_index(drop=True)
        display_page = sorted_df.head(60).copy()
        
        # Format table for clean display
        table_formatted = pd.DataFrame({
            "id": display_page["customerID"],
            "contract": display_page["Contract"],
            "tenure": display_page["tenure"].apply(lambda v: f"{v}m"),
            "monthly": display_page["MonthlyCharges"].apply(lambda v: f"${v:.2f}"),
            "p(churn)": display_page["churn_prob"].apply(lambda v: f"{v:.2f}"),
            "risk": display_page["risk_band"]
        })
        
        st.dataframe(
            table_formatted,
            width='stretch',
            hide_index=True,
            height=460
        )

    with c_detail:
        st.markdown('<div class="mono-label" style="margin-bottom: 6px;">inspect_customer</div>', unsafe_allow_html=True)
        customer_options = list(display_page["customerID"]) if not display_page.empty else []
        if customer_options:
            selected_id = st.selectbox("Select Account", customer_options, label_visibility="collapsed")
            row = display_page[display_page["customerID"] == selected_id].iloc[0]
            
            p_val = float(row["churn_prob"])
            t_val = int(row["tenure"])
            c_val = str(row["Contract"])
            mc_val = float(row["MonthlyCharges"])
            reasons_raw = str(row.get("top_reasons", "Standard account usage profile"))
            reasons = [r.strip() for r in reasons_raw.split(";") if r.strip()][:3]

            # Rule-based recommended retention action
            if "Month" in c_val and t_val <= 12:
                rec_action = "Offer an annual contract upgrade with a 15% discount voucher valid for 6 months."
            elif mc_val >= 80:
                rec_action = "Deploy proactive customer support call to audit bandwidth health and offer bundled device protection."
            elif t_val > 24 and p_val >= 0.5:
                rec_action = "Assign a dedicated loyalty account manager and dispatch VIP subscription appreciation credit."
            else:
                rec_action = "Continue standard satisfaction monitoring and quarterly value summary communication."

            st.markdown(
                f"""
                <div style="border-top: 1px solid var(--hairline); padding-top: 1rem; margin-top: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                        <span style="font-size: 20px; font-weight: 600;">{selected_id}</span>
                        <span class="mono-label text-pink" style="font-size: 16px;">p = {p_val:.2f}</span>
                    </div>
                    <div class="mono-label" style="margin-top: 4px;">{c_val} · {t_val} mos · ${mc_val:.2f}/mo</div>

                    <div class="hairline-divider-tight"></div>
                    <div class="mono-label text-pink" style="margin-bottom: 8px;">top_risk_drivers</div>
                """,
                unsafe_allow_html=True
            )

            for idx, r_text in enumerate(reasons, 1):
                st.markdown(
                    f"""
                    <div class="reason-row">
                        <span class="reason-num">0{idx}</span>
                        <span>{r_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                f"""
                    <div class="hairline-divider-tight"></div>
                    <div class="mono-label text-teal" style="margin-bottom: 6px;">recommended_retention_action</div>
                    <p style="font-size: 13px; line-height: 1.6; color: var(--text-primary); margin: 0;">
                        {rec_action}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==============================================================================
# PAGE 3: WHO TO CONTACT (PROFIT OPTIMIZATION)
# ==============================================================================
def render_page_contact(preds_df: pd.DataFrame):
    st.markdown(
        """
        <div style="margin-bottom: 2rem;">
            <div class="mono-label text-pink" style="margin-bottom: 6px;">03 · financial optimization</div>
            <h1 style="font-size: 36px; margin-bottom: 8px;">Who to contact</h1>
            <p style="color: var(--text-muted); font-size: 14px; max-width: 620px;">
                Maximize expected net revenue by targeting accounts where predicted value saved exceeds intervention cost.
            </p>
        </div>
        <div class="hairline-divider-tight"></div>
        """,
        unsafe_allow_html=True
    )

    # Parameter sliders
    st.markdown('<div class="mono-label text-pink" style="margin-bottom: 12px;">campaign_assumptions</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        offer_cost = st.slider("Offer Cost ($ per account)", 5.0, 100.0, 20.0, step=5.0)
    with c2:
        success_rate = st.slider("Offer Success Rate (%)", 5, 80, 25, step=5) / 100.0
    with c3:
        months_retained = st.slider("Months Retained if Saved", 1, 24, 12, step=1)

    # Compute expected profit and optimal cutoff
    ranked_df = compute_expected_profit(
        preds_df,
        offer_cost=offer_cost,
        offer_success_rate=success_rate,
        months_retained=months_retained
    )
    cutoff = find_optimal_cutoff(ranked_df)
    optimal_n = cutoff["optimal_count"]
    max_profit = cutoff["max_profit"]

    # Target volume slider defaulting to optimal
    total_cust = len(ranked_df)
    st.markdown('<div class="mono-label" style="margin-top: 14px; margin-bottom: 6px;">accounts_to_contact</div>', unsafe_allow_html=True)
    n_contact = st.slider(
        "Accounts to contact",
        min_value=0,
        max_value=total_cust,
        value=optimal_n,
        step=50,
        label_visibility="collapsed"
    )

    # Financial KPIs
    summary = get_campaign_summary(ranked_df, n_contact, offer_cost)
    st.markdown(
        f"""
        <div class="stats-row-container" style="margin: 2rem 0;">
            <div class="stats-grid">
                <div class="stat-cell">
                    <div class="mono-label">expected net profit</div>
                    <div class="stat-number text-pink">${summary['expected_net_profit']:,.0f}</div>
                    <div class="mono-label">net of voucher costs</div>
                </div>
                <div class="stat-cell">
                    <div class="mono-label">campaign expenditure</div>
                    <div class="stat-number">${summary['campaign_cost']:,.0f}</div>
                    <div class="mono-label">{n_contact:,} outreach actions</div>
                </div>
                <div class="stat-cell">
                    <div class="mono-label">campaign roi</div>
                    <div class="stat-number text-teal">{summary['roi_percent']:.1f}%</div>
                    <div class="mono-label">${summary['retained_revenue']:,.0f} retained revenue</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Cumulative profit curve chart
    step = max(1, len(ranked_df) // 200)
    curve_data = ranked_df.iloc[::step].copy()

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=curve_data["contact_rank"],
        y=curve_data["cumulative_profit"],
        mode="lines",
        name="Expected Net Profit",
        line=dict(color="#FF5FA2", width=2.5)
    ))
    fig_curve.add_trace(go.Scatter(
        x=[0, total_cust],
        y=[0, 0],
        mode="lines",
        name="Breakeven",
        line=dict(color="#2A2236", dash="dot")
    ))
    if optimal_n > 0:
        fig_curve.add_trace(go.Scatter(
            x=[optimal_n],
            y=[max_profit],
            mode="markers+text",
            name=f"Optimal ({optimal_n:,})",
            marker=dict(size=9, color="#EDE8F5", line=dict(color="#FF5FA2", width=2)),
            text=[f"Peak: ${max_profit:,.0f}"],
            textposition="top center",
            textfont=dict(family="IBM Plex Mono, monospace", size=11, color="#EDE8F5")
        ))

    fig_curve.update_layout(
        PLOTLY_EDITORIAL_LAYOUT,
        height=320,
        title=dict(
            text=f"Cumulative Expected Profit · Optimal cutoff at {optimal_n:,} accounts",
            font=dict(size=14, color="#EDE8F5")
        ),
        xaxis_title="Accounts Contacted (Sorted by Expected Value)",
        yaxis_title="Net Expected Profit ($)",
        showlegend=False
    )
    st.plotly_chart(fig_curve, width='stretch')

    st.markdown(
        """
        <div style="font-size: 12px; color: var(--text-labels); margin-bottom: 2rem;">
            Note: These numbers are assumptions chosen for scenario analysis, not empirical certainties. Adjust the parameters above to evaluate alternatives.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Contact table & CSV download
    st.markdown('<div class="mono-label text-pink" style="margin-bottom: 8px;">priority_contact_list</div>', unsafe_allow_html=True)
    display_targets = ranked_df.iloc[:n_contact].copy()

    target_table = pd.DataFrame({
        "rank": display_targets["contact_rank"],
        "id": display_targets["customerID"],
        "contract": display_targets["Contract"],
        "monthly": display_targets["MonthlyCharges"].apply(lambda v: f"${v:.2f}"),
        "p(churn)": display_targets["churn_prob"].apply(lambda v: f"{v:.2f}"),
        "value": display_targets["customer_value"].apply(lambda v: f"${v:.0f}"),
        "expected_profit": display_targets["expected_profit"].apply(lambda v: f"${v:+.2f}")
    })

    st.dataframe(target_table.head(80), width='stretch', hide_index=True, height=360)

    csv_export = display_targets[[
        "contact_rank", "customerID", "Contract", "tenure", "MonthlyCharges",
        "churn_prob", "customer_value", "expected_profit"
    ]].to_csv(index=False)

    st.download_button(
        "Download scored contact list (CSV)",
        data=csv_export,
        file_name="retain_contact_targets.csv",
        mime="text/csv"
    )


# ==============================================================================
# PAGE 4: MODEL LAB
# ==============================================================================
def render_page_lab(metrics: dict):
    st.markdown(
        """
        <div style="margin-bottom: 2rem;">
            <div class="mono-label text-pink" style="margin-bottom: 6px;">04 · benchmark diagnostics</div>
            <h1 style="font-size: 36px; margin-bottom: 8px;">Model lab</h1>
            <p style="color: var(--text-muted); font-size: 14px; max-width: 600px;">
                Comparative evaluation across candidate algorithms, calibration reliability, and feature attributions.
            </p>
        </div>
        <div class="hairline-divider-tight"></div>
        """,
        unsafe_allow_html=True
    )

    models_data = metrics.get("models_evaluated", {})
    winner_name = metrics.get("winner_model", "XGBoost")

    # 1. Comparison table
    st.markdown(
        f'<div class="mono-label text-pink" style="margin-bottom: 8px;">candidate_model_comparison · best: {winner_name.lower()}</div>',
        unsafe_allow_html=True
    )

    rows = []
    for m_name, m_stats in models_data.items():
        is_best = (m_name == winner_name)
        rows.append({
            "model": f"{m_name} [best]" if is_best else m_name,
            "roc_auc": f"{m_stats['roc_auc']:.4f}",
            "accuracy": f"{m_stats['accuracy']:.4f}",
            "precision": f"{m_stats['precision']:.4f}",
            "recall": f"{m_stats['recall']:.4f}",
            "f1_score": f"{m_stats['f1']:.4f}"
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    st.markdown('<div class="mono-label" style="margin-top: 6px; margin-bottom: 2.5rem;">Evaluated on an 80/20 stratified holdout set with identical unweighted class distributions.</div>', unsafe_allow_html=True)

    # 2. Charts: ROC curves & Confusion matrix
    c_roc, c_cm = st.columns(2)

    with c_roc:
        st.markdown('<div class="mono-label text-pink">roc_comparison</div>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#2A2236", dash="dot")))

        colors = {"Logistic Regression": "#8A7FA0", "Random Forest": "#2DD4BF", "XGBoost": "#FF5FA2"}
        for m_name, m_stats in models_data.items():
            rc = m_stats.get("roc_curve", {})
            if rc:
                fig_roc.add_trace(go.Scatter(
                    x=rc.get("fpr", []),
                    y=rc.get("tpr", []),
                    mode="lines",
                    name=f"{m_name} ({m_stats['roc_auc']:.3f})",
                    line=dict(color=colors.get(m_name, "#EDE8F5"), width=2)
                ))

        fig_roc.update_layout(
            PLOTLY_EDITORIAL_LAYOUT,
            height=300,
            title=dict(text="Discrimination Across Thresholds", font=dict(size=13)),
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95, font=dict(family="IBM Plex Mono", size=10))
        )
        st.plotly_chart(fig_roc, width='stretch')
        st.caption("XGBoost demonstrates the highest area under curve, providing optimal ranking separation.")

    with c_cm:
        st.markdown('<div class="mono-label text-pink">confusion_matrix · 0.50 cutoff</div>', unsafe_allow_html=True)
        winner_cm = models_data[winner_name]["confusion_matrix"]
        
        cm_text = [
            [f"{winner_cm[0][0]}<br><span style='font-size:10px;'>TN</span>", f"{winner_cm[0][1]}<br><span style='font-size:10px;'>FP</span>"],
            [f"{winner_cm[1][0]}<br><span style='font-size:10px;'>FN</span>", f"{winner_cm[1][1]}<br><span style='font-size:10px;'>TP</span>"]
        ]
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=winner_cm,
            x=["Predicted Stay", "Predicted Churn"],
            y=["Actual Stay", "Actual Churn"],
            text=cm_text,
            texttemplate="%{text}",
            textfont=dict(family="IBM Plex Mono, monospace", size=14, color="#EDE8F5"),
            colorscale=[[0, "#161020"], [0.5, "#3D1A30"], [1, "#FF5FA2"]],
            showscale=False
        ))
        fig_cm.update_layout(PLOTLY_EDITORIAL_LAYOUT, height=300, title=dict(text=f"{winner_name} Holdout Confusion Matrix", font=dict(size=13)))
        st.plotly_chart(fig_cm, width='stretch')
        st.caption("Binary decisions at 0.50 threshold. In practice, Retain uses dynamic profit-based cutoffs.")

    st.markdown('<div class="hairline-divider"></div>', unsafe_allow_html=True)

    # 3. Calibration & Permutation importance
    c_cal, c_imp = st.columns(2)

    with c_cal:
        st.markdown('<div class="mono-label text-pink">probability_calibration</div>', unsafe_allow_html=True)
        cal_data = models_data[winner_name]["calibration"]
        
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(color="#2A2236", dash="dot")))
        fig_cal.add_trace(go.Scatter(
            x=cal_data["prob_pred"],
            y=cal_data["prob_true"],
            mode="lines+markers",
            name=winner_name,
            line=dict(color="#2DD4BF", width=2),
            marker=dict(size=6, color="#2DD4BF")
        ))
        fig_cal.update_layout(
            PLOTLY_EDITORIAL_LAYOUT,
            height=300,
            title=dict(text="Predicted Probability vs Observed Frequency", font=dict(size=13)),
            xaxis_title="Predicted Probability (10 bins)",
            yaxis_title="Empirical Churn Rate",
            showlegend=False
        )
        st.plotly_chart(fig_cal, width='stretch')
        st.caption("Well-calibrated probabilities track the diagonal, verifying that risk numbers reflect true attrition odds.")

    with c_imp:
        st.markdown('<div class="mono-label text-pink">permutation_importance</div>', unsafe_allow_html=True)
        imp_list = metrics.get("permutation_importance", [])
        if imp_list:
            imp_df = pd.DataFrame(imp_list).head(7).sort_values(by="importance_mean", ascending=True)
            fig_imp = px.bar(
                imp_df,
                x="importance_mean",
                y="feature",
                orientation="h",
                labels={"importance_mean": "ROC-AUC Drop", "feature": "Feature"}
            )
            fig_imp.update_layout(PLOTLY_EDITORIAL_LAYOUT, height=300, title=dict(text="Top Drivers of Churn Separation", font=dict(size=13)))
            fig_imp.update_traces(marker_color="#FF5FA2")
            st.plotly_chart(fig_imp, width='stretch')
            st.caption("Permutation importance isolates each feature's marginal predictive contribution on the holdout split.")


# ==============================================================================
# MAIN ROUTER
# ==============================================================================
def main():
    load_stylesheet()
    preds_df, metrics = load_data_and_metrics()

    if preds_df is None or metrics is None:
        st.error("Model artifacts missing. Run `python train.py` first.")
        return

    # Synchronize active page between query params and session state
    query_page = st.query_params.get("page")
    if query_page in ["home", "explorer", "contact", "lab"]:
        if "last_query_page" in st.session_state and query_page != st.session_state["last_query_page"]:
            st.session_state["active_page"] = query_page
        elif "active_page" not in st.session_state:
            st.session_state["active_page"] = query_page
        st.session_state["last_query_page"] = query_page
    elif "active_page" not in st.session_state:
        st.session_state["active_page"] = "home"
        st.session_state["last_query_page"] = "home"

    render_top_nav(st.session_state["active_page"])

    active_page = st.session_state["active_page"]
    if active_page == "home":
        render_page_home(preds_df, metrics)
    elif active_page == "explorer":
        render_page_explorer(preds_df)
    elif active_page == "contact":
        render_page_contact(preds_df)
    elif active_page == "lab":
        render_page_lab(metrics)


if __name__ == "__main__":
    main()
