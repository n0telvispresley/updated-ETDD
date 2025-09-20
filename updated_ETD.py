# updated_ETD_with_escalations_and_optimizer.py
# Full code (original app + zero-counter + optimizer + escalations report)
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io

# Set Streamlit page config
st.set_page_config(page_title="IE Energy Theft Detection Dashboard", layout="wide")

# --------------------------
# Utility functions
# --------------------------
def preserve_exact_string(value):
    if pd.isna(value) or value is None:
        return ""
    return str(value)

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r'\s+', ' ', name.strip().upper())  # Normalize spaces
    name = re.sub(r'[^\w\s-]', '', name)  # Remove special characters except hyphen
    name = re.sub(r'-+', '-', name)  # Replace multiple hyphens with single
    return name

def get_short_name(name, is_dt=False):
    if isinstance(name, str) and name and "-" in name:
        parts = name.split("-")
        return parts[-1].strip()  # Last part for DT or Feeder
    return name if isinstance(name, str) else ""

# --------------------------
# Pattern deviation function (existing)
# --------------------------
def calculate_pattern_deviation(df, id_col, value_cols, score_per_zero=0.2, threshold=0.5):
    pattern_scores = []
    # Ensure only available columns are used
    valid_value_cols = [col for col in value_cols if col in df.columns]
    if not valid_value_cols:
        st.warning(f"No valid value columns in {id_col} dataframe: {value_cols}")
        return pd.DataFrame(columns=["id", "month", "pattern_deviation_score"])
    
    for id_val, group in df.groupby(id_col):
        # Get energy values for the group
        values = group[valid_value_cols].iloc[0].values  # Use iloc[0] for single row per group
        # Map columns to months
        month_map = {col: col.replace(" (kWh)", "") for col in valid_value_cols}
        # Calculate mean of non-zero values
        non_zero_values = values[values > 0]
        mean_non_zero = non_zero_values.mean() if len(non_zero_values) > 0 else 1
        
        # Calculate score for each month
        for idx, col in enumerate(valid_value_cols):
            if idx >= len(values):
                st.warning(f"Index {idx} out of bounds for {id_val} in {id_col}. Skipping.")
                continue
            energy = values[idx]
            score = 0.0
            if energy == 0:
                score += score_per_zero  # Add score for zero reading
            elif energy < mean_non_zero * threshold:
                score += 0.1  # Add score for low reading
            score = min(score, 1.0)  # Cap score at 1.0
            pattern_scores.append({
                "id": id_val,
                "month": month_map.get(col, "Unknown"),
                "pattern_deviation_score": score
            })
    
    result = pd.DataFrame(pattern_scores)
    if result.empty:
        st.warning(f"No pattern deviation scores calculated for {id_col}.")
    return result

# --------------------------
# Zero Counter (new)
# counts zero months across selected months for each customer (normalized 0-1)
# --------------------------
def calculate_zero_counter_from_monthly(customer_monthly, selected_months):
    # customer_monthly expected to have: ACCOUNT_NUMBER, month, billed_kwh
    # We'll compute per-account zero count over selected_months
    sel = customer_monthly[customer_monthly["month"].isin(selected_months)]
    # count number of months considered
    months_count = len(selected_months)
    zero_counts = sel.groupby("ACCOUNT_NUMBER").apply(lambda g: (g["billed_kwh"] == 0).sum()).reset_index(name="zero_count")
    zero_counts["zero_counter_score"] = (zero_counts["zero_count"] / months_count).clip(0,1)
    return zero_counts[["ACCOUNT_NUMBER", "zero_counter_score"]]

# --------------------------
# DT relative usage (existing)
# --------------------------
def calculate_dt_relative_usage(customer_monthly, selected_months):
    # Aggregate customer billed_kwh over selected months
    customer_agg = customer_monthly.groupby(["ACCOUNT_NUMBER", "NAME_OF_DT"])["billed_kwh"].sum().reset_index()
    # Calculate average billed_kwh per DT
    dt_avg = customer_agg.groupby("NAME_OF_DT")["billed_kwh"].mean().reset_index().rename(columns={"billed_kwh": "dt_avg_kwh"})
    customer_agg = customer_agg.merge(dt_avg, on="NAME_OF_DT", how="left")
    # Avoid division by zero in interpolation
    customer_agg["relative_ratio"] = np.where(customer_agg["dt_avg_kwh"] == 0, 0.5, customer_agg["billed_kwh"] / customer_agg["dt_avg_kwh"])
    # Calculate dt_relative_usage_score
    customer_agg["dt_relative_usage_score"] = customer_agg.apply(
        lambda row: 0.9 if row["billed_kwh"] < row["dt_avg_kwh"] * 0.3
        else 0.1 if row["billed_kwh"] > row["dt_avg_kwh"] * 0.7
        else 0.1 + (0.9 - 0.1) * (0.7 - row["relative_ratio"]) / (0.7 - 0.3),
        axis=1
    ).clip(0, 1)
    return customer_agg[["ACCOUNT_NUMBER", "dt_relative_usage_score"]]

# --------------------------
# Escalations report generator
# This function uses preprocessed PPM/PPD (customer_df or aggregated monthly)
# It will return a report DataFrame containing customer name, feeder, DT, monthly energy readings and theft score
# --------------------------
def generate_escalations_report_from_account_summary(account_summary_df, escalations_df, months_list):
    """
    account_summary_df: dataframe indexed by ACCOUNT_NUMBER and contains:
       - CUSTOMER_NAME, NAME_OF_FEEDER, NAME_OF_DT
       - theft_probability (final per-account)
       - monthly columns like JAN (kWh), FEB (kWh), etc or columns with '(kWh)'
    escalations_df: original escalations sheet with 'Account No' column
    """
    escalations_df = escalations_df.copy()
    # Normalize account number field types to str trimmed
    escalations_df['Account No'] = escalations_df['Account No'].astype(str).str.strip()
    account_summary_df = account_summary_df.copy()
    account_summary_df['ACCOUNT_NUMBER'] = account_summary_df['ACCOUNT_NUMBER'].astype(str).str.strip()
    # Prepare columns for monthly readings (we'll use columns named like 'JAN (kWh)' or just 'JAN')
    # For simplicity, we'll include columns in account_summary_df that contain month names from months_list
    month_cols = []
    for m in months_list:
        # possible columns: "M (kWh)" or "M"
        cand1 = f"{m} (kWh)"
        cand2 = m
        if cand1 in account_summary_df.columns:
            month_cols.append(cand1)
        elif cand2 in account_summary_df.columns:
            month_cols.append(cand2)
    # Build a lookup
    account_lookup = account_summary_df.set_index('ACCOUNT_NUMBER')
    report_rows = []
    for acct in escalations_df['Account No']:
        if acct in account_lookup.index:
            row = account_lookup.loc[acct]
            # row may be a Series (single)
            if isinstance(row, pd.Series):
                entry = {
                    "Account No": acct,
                    "Customer Name": row.get("CUSTOMER_NAME", ""),
                    "Feeder": row.get("NAME_OF_FEEDER", "") or row.get("Feeder", ""),
                    "DT": row.get("NAME_OF_DT", ""),
                    "Theft Score": row.get("theft_probability", np.nan)
                }
                # add monthly columns
                for mc in month_cols:
                    entry[mc] = row.get(mc, np.nan)
                report_rows.append(entry)
            else:
                # If multiple rows (shouldn't happen) iterate
                for _, r in row.iterrows():
                    entry = {
                        "Account No": acct,
                        "Customer Name": r.get("CUSTOMER_NAME", ""),
                        "Feeder": r.get("NAME_OF_FEEDER", "") or r.get("Feeder", ""),
                        "DT": r.get("NAME_OF_DT", ""),
                        "Theft Score": r.get("theft_probability", np.nan)
                    }
                    for mc in month_cols:
                        entry[mc] = r.get(mc, np.nan)
                    report_rows.append(entry)
        else:
            report_rows.append({
                "Account No": acct,
                "Customer Name": "NOT FOUND",
                "Feeder": "NOT FOUND",
                "DT": "NOT FOUND",
                "Theft Score": np.nan
            })
    report_df = pd.DataFrame(report_rows)
    return report_df

# --------------------------
# Optimizer for customer-level weights (pattern, relative, zero)
# Uses step 0.05 as requested; normalizes candidate weights among themselves
# Only optimizes pattern, relative, zero; does not alter other weights
# --------------------------
def optimize_customer_level_weights(account_level_df, escalations_df, w_feeder, w_dt, w_location, step=0.05):
    """
    account_level_df must have columns:
      - ACCOUNT_NUMBER
      - feeder_billing_efficiency (0-1)
      - dt_billing_efficiency (0-1)
      - location_trust_score (0-1)
      - pattern_deviation_score (0-1)  (per-account average)
      - dt_relative_usage_score (0-1)
      - zero_counter_score (0-1)
    escalations_df must have 'Account No' column
    Returns best (wp, wr, wz) normalized triple that maximize mean theft prob among escalations accounts.
    """
    # subset escalations accounts
    esc_accounts = set(escalations_df['Account No'].astype(str).str.strip())
    subset = account_level_df[account_level_df['ACCOUNT_NUMBER'].astype(str).str.strip().isin(esc_accounts)].copy()
    if subset.empty:
        return (0.33, 0.33, 0.34)  # fallback equal weights

    best = None
    best_score = -np.inf

    candidates = np.arange(0.0, 1.0 + 1e-9, step)  # include 1.0
    for wp in candidates:
        for wr in candidates:
            for wz in candidates:
                s = wp + wr + wz
                if s == 0:
                    continue
                wp_n, wr_n, wz_n = wp / s, wr / s, wz / s
                # compute candidate theft probability for subset:
                scores = (
                    w_feeder * (1 - subset["feeder_billing_efficiency"]) +
                    w_dt * (1 - subset["dt_billing_efficiency"]) +
                    w_location * subset["location_trust_score"] +
                    wp_n * subset["pattern_deviation_score"] +
                    wr_n * subset["dt_relative_usage_score"] +
                    wz_n * subset["zero_counter_score"]
                )
                mean_score = scores.mean()
                if mean_score > best_score:
                    best_score = mean_score
                    best = (wp_n, wr_n, wz_n)
    # If best is None, fallback to equal
    if best is None:
        best = (0.33, 0.33, 0.34)
    return best

# --------------------------
# --------------------------
# Begin main app code (original code, with additions integrated)
# --------------------------
# --------------------------

# File uploader
st.subheader("Upload Excel File")
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.warning("Please upload an Excel file to proceed.")
    st.stop()

# Read all sheets with converters (preserve string columns exactly)
try:
    sheets = pd.read_excel(
        uploaded_file,
        sheet_name=None,
        converters={
            "Feeder Data": {"Feeder": preserve_exact_string},
            "Transformer Data": {"New Unique DT Nomenclature": preserve_exact_string, "Ownership": preserve_exact_string, "Connection Status": preserve_exact_string},
            "Customer Data_PPM": {"NAME_OF_DT": preserve_exact_string, "NAME_OF_FEEDER": preserve_exact_string, "ACCOUNT_NUMBER": preserve_exact_string, "METER_NUMBER": preserve_exact_string, "BUSINESS_UNIT": preserve_exact_string, "UNDERTAKING": preserve_exact_string, "TARIFF": preserve_exact_string},
            "Customer Data_PPD": {"NAME_OF_DT": preserve_exact_string, "NAME_OF_FEEDER": preserve_exact_string, "ACCOUNT_NUMBER": preserve_exact_string, "METER_NUMBER": preserve_exact_string, "BUSINESS_UNIT": preserve_exact_string, "UNDERTAKING": preserve_exact_string, "TARIFF": preserve_exact_string},
            "Feeder Band": {"BAND": preserve_exact_string, "Feeder": preserve_exact_string, "Short Name": preserve_exact_string},
            "Customer Tariffs": {"Tariff": preserve_exact_string},
            # Make sure to preserve Account No in Escalations
            "Escalations": {"Feeder": preserve_exact_string, "DT Nomenclature": preserve_exact_string, "Account No": preserve_exact_string}
        }
    )
except Exception as e:
    st.error(f"Error reading Excel file: {e}")
    st.stop()

# Access sheets
feeder_df = sheets.get("Feeder Data")
dt_df = sheets.get("Transformer Data")
ppm_df = sheets.get("Customer Data_PPM")
ppd_df = sheets.get("Customer Data_PPD")
band_df = sheets.get("Feeder Band")
tariff_df = sheets.get("Customer Tariffs")
escalations_df = sheets.get("Escalations")

# Verify presence of essential sheets
if any(df is None for df in [feeder_df, dt_df, ppm_df, ppd_df, band_df, tariff_df, escalations_df]):
    st.error("One or more sheets missing. Ensure Excel contains: Feeder Data, Transformer Data, Customer Data_PPM, Customer Data_PPD, Feeder Band, Customer Tariffs, Escalations.")
    st.stop()

# Validate column names (original logic)
required_customer_cols = ["NAME_OF_DT", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "NAME_OF_FEEDER", "BUSINESS_UNIT", "UNDERTAKING", "TARIFF"]
required_dt_cols = ["New Unique DT Nomenclature", "Ownership", "Connection Status"]
required_feeder_cols = ["Feeder"]
required_band_cols = ["Feeder", "BAND"]
required_tariff_cols = ["Tariff"]
required_escalations_cols = ["Feeder", "DT Nomenclature", "Account No"]

for df, name, cols in [
    (ppm_df, "Customer Data_PPM", required_customer_cols),
    (ppd_df, "Customer Data_PPD", required_customer_cols),
    (dt_df, "Transformer Data", required_dt_cols),
    (feeder_df, "Feeder Data", required_feeder_cols),
    (band_df, "Feeder Band", required_band_cols),
    (tariff_df, "Customer Tariffs", required_tariff_cols),
    (escalations_df, "Escalations", required_escalations_cols)
]:
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        if name == "Transformer Data" and "Ownership" in missing_cols:
            st.warning("Ownership column missing in Transformer Data. Assuming all DTs are public.")
            dt_df["Ownership"] = "PUBLIC"
        else:
            st.error(f"Missing columns in {name}: {missing_cols}")
            st.stop()

# Validate month columns
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
month_indices = {month: i for i, month in enumerate(months)}
for df, name in [(ppm_df, "Customer Data_PPM"), (ppd_df, "Customer Data_PPD"), (feeder_df, "Feeder Data"), (dt_df, "Transformer Data")]:
    missing_months = [m for m in months if m not in df.columns]
    if missing_months:
        st.warning(f"Missing month columns in {name}: {missing_months}. Filling with 0.")
        for m in missing_months:
            df[m] = 0

# Handle missing columns defaults
for df, col in [(ppm_df, "TARIFF"), (ppd_df, "TARIFF"), (band_df, "BAND"), (tariff_df, "Tariff")]:
    if col not in df.columns:
        df[col] = ""
if "Short Name" not in band_df.columns:
    band_df["Short Name"] = band_df["Feeder"].apply(lambda x: get_short_name(x))

# Handle Rate column from tariffs
rate_col = next((col for col in ["Rate (NGN)", "Rate (₦)", "Rate", "RATE", "Rate(NGN)", "Rate(₦)"] if col in tariff_df.columns), None)
if rate_col:
    tariff_df["Rate (NGN)"] = pd.to_numeric(tariff_df[rate_col], errors="coerce").fillna(209.5)
else:
    tariff_df["Rate (NGN)"] = 209.5

# Compute band-specific tariff rates
band_tariffs = {
    "A": ["A-MD1", "A-MD2", "A-Non MD"],
    "B": ["B-MD1", "B-MD2", "B-Non MD"],
    "C": ["C-MD1", "C-MD2", "C-Non MD"],
    "D": ["D-MD1", "D-MD2", "D-Non MD"],
    "E": ["E-MD1", "E-MD2", "E-Non MD"]
}
band_rates = {}
for band, tariffs in band_tariffs.items():
    rates = tariff_df[tariff_df["Tariff"].isin(tariffs)]["Rate (NGN)"]
    band_rates[band] = rates.mean() if not rates.empty else 209.5

# Map feeders to bands
feeder_df = feeder_df.merge(band_df[["Feeder", "BAND"]], on="Feeder", how="left")
feeder_df["BAND"] = feeder_df["BAND"].fillna("Unknown")
feeder_df["Tariff_Rate"] = feeder_df["BAND"].map(band_rates).fillna(209.5)
if feeder_df["BAND"].str.contains("Unknown").any():
    st.warning("Some feeders not mapped to bands. Using default tariff rate (209.5 NGN/kWh).")

# Data preprocessing: convert month columns to kWh format columns
for month in months:
    for df, unit in [(feeder_df, 1000), (ppm_df, 1), (ppd_df, 1)]:
        col = f"{month} (kWh)"
        if month in df.columns:
            df[col] = pd.to_numeric(df[month], errors="coerce").fillna(0) * unit
        else:
            df[col] = 0
    if month in dt_df.columns:
        dt_df[f"{month} (kWh)"] = pd.to_numeric(dt_df[month], errors="coerce").fillna(0)
    else:
        dt_df[f"{month} (kWh)"] = 0

# Drop original month columns to avoid duplication
for df in [feeder_df, dt_df, ppm_df, ppd_df]:
    df.drop(columns=months, errors="ignore", inplace=True)

# Filter NOT CONNECTED DTs with zero energy
dt_df["total_energy_kwh"] = dt_df[[f"{m} (kWh)" for m in months]].sum(axis=1)
not_connected_zero = (dt_df["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_df["total_energy_kwh"] == 0)
excluded_dts = dt_df[not_connected_zero][["New Unique DT Nomenclature", "Connection Status", "total_energy_kwh"]]
if not excluded_dts.empty:
    st.warning(f"Excluding {len(excluded_dts)} DTs marked 'NOT CONNECTED' with zero energy across all months.")
dt_df = dt_df[~not_connected_zero]

# Normalize names
for col, df in [
    ("Feeder", feeder_df), ("NAME_OF_FEEDER", ppm_df), ("NAME_OF_FEEDER", ppd_df), ("Feeder", band_df),
    ("NAME_OF_DT", ppm_df), ("NAME_OF_DT", ppd_df), ("New Unique DT Nomenclature", dt_df),
    ("TARIFF", ppm_df), ("TARIFF", ppd_df), ("Tariff", tariff_df),
    ("BUSINESS_UNIT", ppm_df), ("BUSINESS_UNIT", ppd_df),
    ("UNDERTAKING", ppm_df), ("UNDERTAKING", ppd_df),
    ("Feeder", escalations_df), ("DT Nomenclature", escalations_df)
]:
    if col in df.columns:
        df[col] = df[col].apply(normalize_name)

# Combine PPM and PPD
ppm_df["Billing_Type"] = "PPM"
ppd_df["Billing_Type"] = "PPD"
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True)
if customer_df.empty:
    st.error("customer_df is empty.")
    st.stop()

# Filter for valid DTs from Transformer Data
valid_dts = set(dt_df["New Unique DT Nomenclature"])
customer_invalid_dts = customer_df[~customer_df["NAME_OF_DT"].isin(valid_dts)]
error_report = []
if not customer_invalid_dts.empty:
    for _, row in customer_invalid_dts.iterrows():
        error_report.append({
            "ACCOUNT_NUMBER": row["ACCOUNT_NUMBER"],
            "NAME_OF_DT": row["NAME_OF_DT"],
            "NAME_OF_FEEDER": row["NAME_OF_FEEDER"],
            "BUSINESS_UNIT": row["BUSINESS_UNIT"],
            "UNDERTAKING": row["UNDERTAKING"],
            "Reason": "NAME_OF_DT not in Transformer Data"
        })
error_report_df = pd.DataFrame(error_report)
customer_df = customer_df[customer_df["NAME_OF_DT"].isin(valid_dts)]
if customer_df.empty:
    st.error("No valid customers after filtering for Transformer Data DTs.")
    st.stop()

# Create short names and feeder links
feeder_df["Feeder_Short"] = feeder_df["Feeder"].apply(lambda x: get_short_name(x))
dt_df["DT_Short_Name"] = dt_df["New Unique DT Nomenclature"].apply(lambda x: get_short_name(x, is_dt=True))
dt_df["Feeder"] = dt_df["New Unique DT Nomenclature"].apply(
    lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x and len(x.split("-")) >= 3 else x
)
dt_df["Feeder"] = dt_df["Feeder"].apply(normalize_name)
dt_df["NAME_OF_DT"] = dt_df["New Unique DT Nomenclature"]
customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))
customer_df["Feeder"] = customer_df["NAME_OF_DT"].apply(
    lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x and len(x.split("-")) >= 3 else x
)
customer_df["Feeder"] = customer_df["Feeder"].apply(normalize_name)

# Filter for valid feeders
valid_feeders = set(feeder_df["Feeder"])
dt_df = dt_df[dt_df["Feeder"].isin(valid_feeders)]
customer_df = customer_df[customer_df["Feeder"].isin(valid_feeders)]
if dt_df.empty or customer_df.empty:
    st.error("No valid data after filtering for Feeder Data feeders.")
    st.stop()

# Map DTs to feeder tariff rates
dt_df = dt_df.merge(feeder_df[["Feeder", "Tariff_Rate"]], on="Feeder", how="left")
dt_df["Tariff_Rate"] = dt_df["Tariff_Rate"].fillna(209.5)

# Merge tariffs
tariff_matches = customer_df["TARIFF"].isin(tariff_df["Tariff"])
if not tariff_matches.all():
    st.warning(f"Some TARIFF values in customer data not found in Customer Tariffs: {customer_df[~tariff_matches]['TARIFF'].unique()}")
customer_df = customer_df.merge(tariff_df[["Tariff", "Rate (NGN)"]], left_on="TARIFF", right_on="Tariff", how="left")
customer_df["Rate (NGN)"] = customer_df["Rate (NGN)"].fillna(209.5)
customer_df = customer_df.drop(columns=["Tariff"], errors="ignore")

# Calculate location trust scores from Escalations
escalations_df["Report_Count"] = 1  # Each row is one report
feeder_escalations = escalations_df.groupby("Feeder")["Report_Count"].sum().reset_index()
if not feeder_escalations["Report_Count"].empty:
    feeder_escalations["location_trust_score"] = feeder_escalations["Report_Count"] / feeder_escalations["Report_Count"].max()
else:
    feeder_escalations["location_trust_score"] = 0
feeder_escalations["location_trust_score"] = feeder_escalations["location_trust_score"].fillna(0).clip(0, 1)

dt_escalations = escalations_df.groupby("DT Nomenclature")["Report_Count"].sum().reset_index()
if not dt_escalations["Report_Count"].empty:
    dt_escalations["location_trust_score"] = dt_escalations["Report_Count"] / dt_escalations["Report_Count"].max()
else:
    dt_escalations["location_trust_score"] = 0
dt_escalations["location_trust_score"] = dt_escalations["location_trust_score"].fillna(0).clip(0, 1)

# --------------------------
# Streamlit UI: Filters (left as original)
# --------------------------
st.title("Ikeja Electric Energy Theft Detection Dashboard")
st.subheader("Filters")
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    bu_options = sorted(customer_df["BUSINESS_UNIT"].unique())
    if bu_options:
        selected_bu = st.selectbox("Select Business Unit", bu_options)
    else:
        selected_bu = ""
        st.warning("No Business Units available.")
with col2:
    if selected_bu:
        customer_df_bu = customer_df[customer_df["BUSINESS_UNIT"] == selected_bu]
        ut_options = sorted(customer_df_bu["UNDERTAKING"].unique())
        if ut_options:
            selected_ut = st.selectbox("Select Undertaking", ut_options)
        else:
            selected_ut = ""
            st.warning("No Undertakings available for selected BU.")
    else:
        customer_df_bu = pd.DataFrame()
        selected_ut = ""
with col3:
    if selected_ut:
        customer_df_ut = customer_df_bu[customer_df_bu["UNDERTAKING"] == selected_ut]
        feeder_options = sorted(feeder_df["Feeder_Short"].unique())
        if feeder_options:
            selected_feeder_short = st.selectbox("Select Feeder", feeder_options)
        else:
            selected_feeder_short = ""
            st.error("No feeders available in Feeder Data.")
            st.stop()
    else:
        customer_df_ut = pd.DataFrame()
        selected_feeder_short = ""
with col4:
    if selected_feeder_short:
        selected_feeder = feeder_df[feeder_df["Feeder_Short"] == selected_feeder_short]["Feeder"].iloc[0]
        dt_df_filtered = dt_df[dt_df["Feeder"] == selected_feeder]
        dt_options = sorted(dt_df_filtered["DT_Short_Name"].unique())
        if dt_options:
            selected_dt_short = st.selectbox("Select DT", dt_options)
        else:
            selected_dt_short = ""
            st.error(f"No DTs available for feeder {selected_feeder_short}.")
            st.stop()
    else:
        selected_feeder = ""
        selected_dt_short = ""
with col5:
    start_month = st.selectbox("Start Month", months)
with col6:
    end_month = st.selectbox("End Month", months, index=len(months)-1)
    if month_indices[start_month] > month_indices[end_month]:
        st.error("Start Month must be before or equal to End Month.")
        st.stop()

# --------------------------
# Dynamic Weight Sliders (including new zero counter weight)
# Use keys so we can update via st.session_state after optimization
# --------------------------
st.subheader("Adjust Theft Probability Weights")
colw1, colw2, colw3 = st.columns(3)
with colw1:
    if "w_feeder" not in st.session_state: st.session_state["w_feeder"] = 0.2
    w_feeder = st.slider("Feeder Billing Efficiency Weight", 0.0, 1.0, st.session_state["w_feeder"], 0.01, key="w_feeder")
    if "w_dt" not in st.session_state: st.session_state["w_dt"] = 0.2
    w_dt = st.slider("DT Billing Efficiency Weight", 0.0, 1.0, st.session_state["w_dt"], 0.01, key="w_dt")
with colw2:
    if "w_location" not in st.session_state: st.session_state["w_location"] = 0.2
    w_location = st.slider("Location Trust Score Weight", 0.0, 1.0, st.session_state["w_location"], 0.01, key="w_location")
    if "w_pattern" not in st.session_state: st.session_state["w_pattern"] = 0.2
    w_pattern = st.slider("Consumption Pattern Deviation Weight", 0.0, 1.0, st.session_state["w_pattern"], 0.01, key="w_pattern")
with colw3:
    if "w_relative" not in st.session_state: st.session_state["w_relative"] = 0.2
    w_relative = st.slider("DT Relative Usage Score Weight", 0.0, 1.0, st.session_state["w_relative"], 0.01, key="w_relative")
    if "w_zero" not in st.session_state: st.session_state["w_zero"] = 0.2
    w_zero = st.slider("Zero Counter Weight", 0.0, 1.0, st.session_state["w_zero"], 0.01, key="w_zero")

# Normalize weights for display calculation check (they will be normalized later before final calc)
total_weight = w_feeder + w_dt + w_location + w_pattern + w_relative + w_zero
if total_weight == 0:
    st.error("Total weight cannot be zero. Please adjust weights.")
    st.stop()

# --------------------------
# Prepare monthly customer_monthly (melt)
# --------------------------
required_id_vars = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type", "Feeder", "Rate (NGN)"]
value_vars = [f"{m} (kWh)" for m in months]
missing_id_vars = [col for col in required_id_vars if col not in customer_df.columns]
if missing_id_vars:
    st.error(f"Missing id_vars in customer_df: {missing_id_vars}")
    st.stop()
try:
    customer_monthly = customer_df.melt(id_vars=required_id_vars, value_vars=value_vars, var_name="month", value_name="billed_kwh")
    customer_monthly["month"] = customer_monthly["month"].str.replace(" (kWh)", "")
    customer_monthly["month"] = pd.Categorical(customer_monthly["month"], categories=months, ordered=True)
except Exception as e:
    st.error(f"Melt failed: {e}")
    st.stop()

# DT consumption (per month for heatmap)
try:
    dt_agg_monthly = dt_df.melt(id_vars=["New Unique DT Nomenclature", "DT_Short_Name", "Feeder", "Tariff_Rate", "Ownership", "Connection Status", "total_energy_kwh"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="total_dt_kwh")
    dt_agg_monthly["month"] = dt_agg_monthly["month"].str.replace(" (kWh)", "")
    dt_agg_monthly["month"] = pd.Categorical(dt_agg_monthly["month"], categories=months, ordered=True)
except Exception as e:
    st.error(f"DT melt failed: {e}")
    st.stop()

# Filter by month range
selected_months = months[month_indices[start_month]:month_indices[end_month] + 1]
if not selected_months:
    st.error("No months selected.")
    st.stop()
customer_monthly = customer_monthly[customer_monthly["month"].isin(selected_months)]
dt_agg_monthly = dt_agg_monthly[dt_agg_monthly["month"].isin(selected_months)]
if customer_monthly.empty or dt_agg_monthly.empty:
    st.error(f"No data for selected months {selected_months}.")
    st.stop()

# Calculate pattern deviation scores per customer and per DT (existing)
customer_pattern = calculate_pattern_deviation(
    customer_df,
    id_col="ACCOUNT_NUMBER",
    value_cols=[f"{m} (kWh)" for m in months]
)
dt_pattern = calculate_pattern_deviation(
    dt_df,
    id_col="New Unique DT Nomenclature",
    value_cols=[f"{m} (kWh)" for m in months]
)

# Calculate zero_counter using the selected months (new)
zero_counts = calculate_zero_counter_from_monthly(customer_monthly, selected_months)

# Calculate DT relative usage score
dt_relative_usage = calculate_dt_relative_usage(customer_monthly, selected_months)

# Aggregate customer data over month range: customer_agg
period_label = f"{start_month}" if start_month == end_month else f"{start_month} to {end_month}"
try:
    customer_agg = customer_monthly.groupby(["NAME_OF_DT", "Feeder"])["billed_kwh"].sum().reset_index()
    customer_agg.rename(columns={"billed_kwh": "customer_billed_kwh"}, inplace=True)
except Exception as e:
    st.error(f"Customer aggregation failed: {e}")
    st.stop()

# Aggregate DT data over month range
try:
    dt_agg_sum = dt_agg_monthly.groupby(["New Unique DT Nomenclature", "DT_Short_Name", "Feeder", "Tariff_Rate", "Ownership", "Connection Status", "total_energy_kwh"])["total_dt_kwh"].sum().reset_index()
except Exception as e:
    st.error(f"DT aggregation failed: {e}")
    st.stop()

# DT billing efficiency
try:
    dt_merged = dt_agg_sum.merge(customer_agg, left_on=["New Unique DT Nomenclature", "Feeder"], right_on=["NAME_OF_DT", "Feeder"], how="left")
    dt_merged["customer_billed_kwh"] = dt_merged["customer_billed_kwh"].fillna(0)
    dt_merged["total_billed_kwh"] = np.where(
        dt_merged["Ownership"].str.strip().str.upper().isin(["PRIVATE"]),
        dt_merged["total_dt_kwh"],
        dt_merged["customer_billed_kwh"]
    )
    dt_merged["dt_billing_efficiency"] = np.where(
        (dt_merged["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_merged["total_energy_kwh"] > 0),
        0.0,  # High theft risk for NOT CONNECTED with energy
        (dt_merged["total_billed_kwh"] / dt_merged["total_dt_kwh"].replace(0, 1)).clip(0, 1)
    )
    dt_merged["energy_lost_kwh"] = dt_merged["total_dt_kwh"] - dt_merged["total_billed_kwh"]
    dt_merged["financial_loss_naira"] = dt_merged["energy_lost_kwh"] * dt_merged["Tariff_Rate"]
except Exception as e:
    st.error(f"DT merge failed: {e}")
    st.stop()

# Per-month DT billing efficiency for heatmap
try:
    customer_billed_monthly = customer_monthly.groupby(["NAME_OF_DT", "month"])["billed_kwh"].sum().reset_index().rename(columns={"billed_kwh": "customer_billed_kwh"})
    dt_merged_monthly = dt_agg_monthly.merge(customer_billed_monthly, 
                                            left_on=["New Unique DT Nomenclature", "month"], 
                                            right_on=["NAME_OF_DT", "month"], 
                                            how="left")
    dt_merged_monthly["customer_billed_kwh"] = dt_merged_monthly["customer_billed_kwh"].fillna(0)
    dt_merged_monthly["total_billed_kwh"] = np.where(
        dt_merged_monthly["Ownership"].str.strip().str.upper().isin(["PRIVATE"]),
        dt_merged_monthly["total_dt_kwh"],
        dt_merged_monthly["customer_billed_kwh"]
    )
    dt_merged_monthly["dt_billing_efficiency"] = np.where(
        (dt_merged_monthly["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_merged_monthly["total_energy_kwh"] > 0),
        0.0,
        (dt_merged_monthly["total_billed_kwh"] / dt_merged_monthly["total_dt_kwh"].replace(0, 1)).clip(0, 1)
    )
except Exception as e:
    st.error(f"DT monthly merge failed: {e}")
    st.stop()

# Feeder consumption
try:
    feeder_monthly = feeder_df.melt(id_vars=["Feeder", "Feeder_Short", "Tariff_Rate"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="feeder_energy_kwh")
    feeder_monthly["month"] = feeder_monthly["month"].str.replace(" (kWh)", "")
    feeder_monthly["month"] = pd.Categorical(feeder_monthly["month"], categories=months, ordered=True)
    feeder_monthly = feeder_monthly[feeder_monthly["month"].isin(selected_months)]
    feeder_agg = feeder_monthly.groupby(["Feeder", "Feeder_Short", "Tariff_Rate"])["feeder_energy_kwh"].sum().reset_index()
except Exception as e:
    st.error(f"Feeder melt failed: {e}")
    st.stop()

# Feeder billing efficiency
try:
    feeder_agg_billed = dt_merged.groupby(["Feeder"])["total_billed_kwh"].sum().reset_index()
    feeder_merged = feeder_agg.merge(feeder_agg_billed, on=["Feeder"], how="left")
    feeder_merged["total_billed_kwh"] = feeder_merged["total_billed_kwh"].fillna(0)
    feeder_merged["feeder_billing_efficiency"] = (feeder_merged["total_billed_kwh"] / feeder_merged["feeder_energy_kwh"].replace(0, 1)).clip(0, 1)
    feeder_merged["feeder_energy_lost_kwh"] = feeder_merged["feeder_energy_kwh"] - feeder_merged["total_billed_kwh"]
    feeder_merged["feeder_financial_loss_naira"] = feeder_merged["feeder_energy_lost_kwh"] * feeder_merged["Tariff_Rate"]
except Exception as e:
    st.error(f"Feeder merge failed: {e}")
    st.stop()

# Merge location trust scores
feeder_merged = feeder_merged.merge(feeder_escalations[["Feeder", "location_trust_score"]], on="Feeder", how="left")
feeder_merged["location_trust_score"] = feeder_merged["location_trust_score"].fillna(0)
dt_merged = dt_merged.merge(dt_escalations[["DT Nomenclature", "location_trust_score"]], left_on="New Unique DT Nomenclature", right_on="DT Nomenclature", how="left")
dt_merged["location_trust_score"] = dt_merged["location_trust_score"].fillna(0)
dt_merged_monthly = dt_merged_monthly.merge(dt_escalations[["DT Nomenclature", "location_trust_score"]], left_on="New Unique DT Nomenclature", right_on="DT Nomenclature", how="left")
dt_merged_monthly["location_trust_score"] = dt_merged_monthly["location_trust_score"].fillna(0)

# --------------------------
# Prepare account-level summary for optimizer & escalations report
# We'll compute account-level averages of the per-month metrics
# --------------------------
# First, ensure customer_monthly has feeder_billing_efficiency, dt_billing_efficiency, location_trust_score, pattern and dt_relative
customer_monthly = customer_monthly.merge(feeder_merged[["Feeder", "feeder_billing_efficiency", "location_trust_score"]], on="Feeder", how="left")
customer_monthly = customer_monthly.merge(dt_merged_monthly[["New Unique DT Nomenclature", "month", "dt_billing_efficiency", "location_trust_score"]], left_on=["NAME_OF_DT", "month"], right_on=["New Unique DT Nomenclature", "month"], how="left")

# Merge pattern deviation per month for customer
customer_monthly = customer_monthly.merge(customer_pattern, left_on=["ACCOUNT_NUMBER", "month"], right_on=["id", "month"], how="left").drop(columns=["id"], errors="ignore")

# Merge dt_relative_usage (per-account)
customer_monthly = customer_monthly.merge(dt_relative_usage, on="ACCOUNT_NUMBER", how="left")

# Merge zero counter (per-account)
customer_monthly = customer_monthly.merge(zero_counts, on="ACCOUNT_NUMBER", how="left")
customer_monthly["zero_counter_score"] = customer_monthly["zero_counter_score"].fillna(0)

# Fill NA for merged columns
for col in ["feeder_billing_efficiency", "dt_billing_efficiency"]:
    if col in customer_monthly.columns:
        customer_monthly[col] = customer_monthly[col].fillna(0)
if "location_trust_score_x" in customer_monthly.columns and "location_trust_score_y" in customer_monthly.columns:
    customer_monthly["location_trust_score"] = customer_monthly["location_trust_score_x"].combine_first(customer_monthly["location_trust_score_y"]).fillna(0)
elif "location_trust_score" not in customer_monthly.columns:
    customer_monthly["location_trust_score"] = 0

customer_monthly["pattern_deviation_score"] = customer_monthly["pattern_deviation_score"].fillna(0)
customer_monthly["dt_relative_usage_score"] = customer_monthly["dt_relative_usage_score"].fillna(0)

# Build account-level summary (for optimizer & escalations lookup)
account_level = customer_monthly.groupby("ACCOUNT_NUMBER").agg({
    "CUSTOMER_NAME": "first",
    "Feeder": "first",
    "NAME_OF_DT": "first",
    "feeder_billing_efficiency": "mean",
    "dt_billing_efficiency": "mean",
    "location_trust_score": "mean",
    "pattern_deviation_score": "mean",
    "dt_relative_usage_score": "mean",
    "zero_counter_score": "mean"
}).reset_index()

# --------------------------
# Optimization button (optimize pattern/relative/zero for escalations)
# Use step 0.05 as requested, and only optimize customer-level weights
# --------------------------
st.subheader("Optimization (optional)")
if st.button("Optimize Customer-Level Weights for Escalations (step=0.05)"):
    wp_n, wr_n, wz_n = optimize_customer_level_weights(account_level, escalations_df, w_feeder, w_dt, w_location, step=0.05)
    # update session_state sliders so the UI reflects the optimized values for customer-level weights
    st.session_state["w_pattern"] = float(wp_n)
    st.session_state["w_relative"] = float(wr_n)
    st.session_state["w_zero"] = float(wz_n)
    st.success(f"Optimized (pattern, relative, zero): {wp_n:.3f}, {wr_n:.3f}, {wz_n:.3f}")

# After potential optimizer update, read weights from session_state to use in final calc
w_feeder = st.session_state.get("w_feeder", w_feeder)
w_dt = st.session_state.get("w_dt", w_dt)
w_location = st.session_state.get("w_location", w_location)
w_pattern = st.session_state.get("w_pattern", w_pattern)
w_relative = st.session_state.get("w_relative", w_relative)
w_zero = st.session_state.get("w_zero", w_zero)

# Normalize weights so they sum to 1 (to produce theft_probability between 0 and 1)
total_w = w_feeder + w_dt + w_location + w_pattern + w_relative + w_zero
if total_w == 0:
    st.error("Total weight cannot be zero (after normalization). Adjust sliders.")
    st.stop()
w_feeder /= total_w
w_dt /= total_w
w_location /= total_w
w_pattern /= total_w
w_relative /= total_w
w_zero /= total_w

# --------------------------
# Compute theft probability per customer-month (final)
# --------------------------
customer_monthly["theft_probability"] = (
    w_feeder * (1 - customer_monthly["feeder_billing_efficiency"]) +
    w_dt * (1 - customer_monthly["dt_billing_efficiency"]) +
    w_location * customer_monthly["location_trust_score"] +
    w_pattern * customer_monthly["pattern_deviation_score"] +
    w_relative * customer_monthly["dt_relative_usage_score"] +
    w_zero * customer_monthly["zero_counter_score"]
).clip(0, 1)

# Compute per-account summarised theft_probability and risk tier
account_summary = customer_monthly.groupby("ACCOUNT_NUMBER").agg({
    "CUSTOMER_NAME": "first",
    "Feeder": "first",
    "NAME_OF_DT": "first",
    "METER_NUMBER": "first",
    "Billing_Type": "first",
    "billed_kwh": "sum",
    "theft_probability": "mean"
}).reset_index().rename(columns={"billed_kwh": "total_billed_kwh"})

# Risk tier
account_summary["risk_tier"] = pd.cut(account_summary["theft_probability"], bins=[-1e-9, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"], include_lowest=True)

# --------------------------
# UI: Escalations Report (independent of filters)
# --------------------------
st.subheader("Escalations Report (cross-reference Escalations -> PPM/PPD)")
st.markdown("This report checks every `Account No` in the Escalations sheet, finds the customer in PPM/PPD (if present), and returns the customer's monthly readings and final theft probability.")

# Prepare account_summary_with_months (so monthly columns are present)
# Create a pivot for monthly readings per account for the selected months
pivot_monthly = customer_monthly.pivot_table(index="ACCOUNT_NUMBER", columns="month", values="billed_kwh", aggfunc="sum")
# rename month columns to include ' (kWh)' to match original naming if desired
pivot_monthly.columns = [f"{c} (kWh)" for c in pivot_monthly.columns]

# merge pivot into account_summary
account_summary_with_months = account_summary.merge(pivot_monthly.reset_index(), on="ACCOUNT_NUMBER", how="left")

# When Generate button pressed, create report and allow download
if st.button("Generate Escalations Report"):
    report_df = generate_escalations_report_from_account_summary(account_summary_with_months, escalations_df, months)
    st.success(f"Escalations report generated: {len(report_df)} rows (includes NOT FOUND entries).")
    st.dataframe(report_df, use_container_width=True)

    # Provide Excel download
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        report_df.to_excel(writer, sheet_name="Escalations Report", index=False)
    buffer.seek(0)
    st.download_button(
        label="📥 Download Escalations Report (.xlsx)",
        data=buffer.getvalue(),
        file_name=f"Escalations_Report_{period_label.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# --------------------------
# Keep the rest of your original UI, heatmaps, tables, export functionality intact
# (Below are examples of how you might continue using customer_monthly, dt_merged_monthly, feeder_merged, etc.)
# For brevity I will provide the existing Customer List & CSV export logic using account_summary as before
# --------------------------

st.subheader(f"Customers under {selected_dt_short} ({selected_feeder_short}, {period_label})")
if selected_dt_short and st.button("Show Customer List"):
    try:
        filtered_customers = customer_monthly[customer_monthly["DT_Short_Name"] == selected_dt_short]
        if filtered_customers.empty:
            st.warning("No customers for this DT.")
        else:
            month_customers = filtered_customers.groupby(["ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type"]).agg({
                "billed_kwh": "sum",
                "theft_probability": "mean",
                "pattern_deviation_score": "mean",
                "dt_relative_usage_score": "mean",
                "zero_counter_score": "mean"
            }).reset_index()
            display_columns = ["ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "billed_kwh", "Billing_Type", "theft_probability", "pattern_deviation_score", "dt_relative_usage_score", "zero_counter_score"]
            missing_cols = [col for col in display_columns if col not in month_customers.columns]
            if missing_cols:
                st.error(f"Missing columns in month_customers: {missing_cols}")
            else:
                month_customers = month_customers.sort_values(by="theft_probability", ascending=False)
                styled_df = month_customers[display_columns].style.format({
                    "billed_kwh": "{:.2f}",
                    "theft_probability": "{:.3f}",
                    "pattern_deviation_score": "{:.3f}",
                    "dt_relative_usage_score": "{:.3f}",
                    "zero_counter_score": "{:.3f}"
                }).highlight_max(subset=["theft_probability"], color="lightcoral")
                st.dataframe(styled_df)
    except Exception as e:
        st.error(f"Customer list failed: {e}")

# CSV Export for displayed customers (account_summary)
st.subheader("Export Customer Data")
if 'account_summary' in locals() and not account_summary.empty:
    csv = account_summary.to_csv(index=False)
    st.download_button(label=f"Download Account Summary ({period_label})", data=csv, file_name=f"theft_analysis_accounts_{period_label.replace(' ', '_')}.csv", mime="text/csv")
else:
    st.info("No account summary to export.")

# Footer
st.markdown("Built by Elvis Ebenuwah for Ikeja Electric. 2025.")
