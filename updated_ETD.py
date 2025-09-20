# updated_ETD_with_escalations_and_optimizer.py
import io
import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="IE Energy Theft Detection Dashboard", layout="wide")

# ------------------------
# Utility functions
# ------------------------
def preserve_exact_string(value):
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip()

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r'\s+', ' ', name.strip().upper())
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'-+', '-', name)
    return name

def get_short_name(name, is_dt=False):
    if isinstance(name, str) and name and "-" in name:
        parts = name.split("-")
        return parts[-1].strip()
    return name if isinstance(name, str) else ""

# ------------------------
# Scoring functions
# ------------------------
def calculate_pattern_deviation(wide_df, id_col, value_cols):
    """
    For each id (customer), calculate fraction of months where reading < 60% of customer's max non-zero month.
    If all months are zero -> return 1.0 (suspicious).
    """
    results = []
    valid_value_cols = [c for c in value_cols if c in wide_df.columns]
    if not valid_value_cols:
        return pd.DataFrame(columns=["id", "pattern_deviation_score"])
    for id_val, group in wide_df.groupby(id_col):
        values = group[valid_value_cols].iloc[0].values.astype(float)
        nonzero = values[values > 0]
        if len(nonzero) == 0:
            score = 1.0
        else:
            max_val = nonzero.max()
            # Count *all months* (including zeros) below 60% of max
            below_count = np.sum(values < 0.6 * max_val)
            score = below_count / len(valid_value_cols)
        results.append({"id": id_val, "pattern_deviation_score": float(min(score, 1.0))})
    return pd.DataFrame(results)

def calculate_zero_counter(wide_df, id_col, value_cols):
    """
    Fraction of months with zero reading for each id.
    """
    results = []
    valid_value_cols = [c for c in value_cols if c in wide_df.columns]
    if not valid_value_cols:
        return pd.DataFrame(columns=["id", "zero_counter_score"])
    for id_val, group in wide_df.groupby(id_col):
        values = group[valid_value_cols].iloc[0].values.astype(float)
        zeros = np.sum(values == 0)
        score = zeros / len(valid_value_cols)
        results.append({"id": id_val, "zero_counter_score": float(min(score, 1.0))})
    return pd.DataFrame(results)

def calculate_dt_relative_usage(customer_monthly):
    """
    Aggregate billed_kwh per ACCOUNT_NUMBER over selected months -> then compute DT average ignoring zero totals.
    Returns per-account dt_relative_usage_score.
    """
    if customer_monthly.empty:
        return pd.DataFrame(columns=["ACCOUNT_NUMBER", "dt_relative_usage_score"])
    customer_agg = customer_monthly.groupby(["ACCOUNT_NUMBER", "NAME_OF_DT"], dropna=False)["billed_kwh"].sum().reset_index()
    # Compute DT averages ignoring customers with zero billed_kwh
    nonzero_customer_agg = customer_agg[customer_agg["billed_kwh"] > 0]
    dt_avg = nonzero_customer_agg.groupby("NAME_OF_DT")["billed_kwh"].mean().reset_index().rename(columns={"billed_kwh": "dt_avg_kwh"})
    customer_agg = customer_agg.merge(dt_avg, on="NAME_OF_DT", how="left")
    customer_agg["dt_avg_kwh"] = customer_agg["dt_avg_kwh"].fillna(0)
    customer_agg["relative_ratio"] = np.where(customer_agg["dt_avg_kwh"] == 0, 0.5, customer_agg["billed_kwh"] / customer_agg["dt_avg_kwh"])
    # Interpolated score
    def score_row(row):
        if row["dt_avg_kwh"] == 0:
            # If DT avg is zero (no activity), return moderate score based on customer's billed_kwh being zero or not
            return 0.9 if row["billed_kwh"] == 0 else 0.1
        if row["billed_kwh"] < row["dt_avg_kwh"] * 0.3:
            return 0.9
        if row["billed_kwh"] > row["dt_avg_kwh"] * 0.7:
            return 0.1
        # linear interpolation between 0.3 and 0.7
        return 0.1 + (0.9 - 0.1) * (0.7 - row["relative_ratio"]) / (0.7 - 0.3)
    customer_agg["dt_relative_usage_score"] = customer_agg.apply(score_row, axis=1).clip(0, 1)
    return customer_agg[["ACCOUNT_NUMBER", "dt_relative_usage_score"]]

# ------------------------
# Optimizer
# ------------------------
def optimize_customer_weights_for_escalations(customers_features_df, escalations_accounts, w_feeder, w_dt, w_location, step=0.05):
    """
    Optimize the relative weights assigned to: pattern_deviation, dt_relative_usage, zero_counter
    to maximize mean theft probability among escalation accounts using current fixed weights for feeder, dt, location.
    Returns (best_wp, best_wr, best_wz), pre_mean, best_mean
    """
    # customers_features_df: index ACCOUNT_NUMBER or column ACCOUNT_NUMBER and contains the required features:
    # feeder_billing_efficiency, dt_billing_efficiency, location_trust_score, pattern_deviation_score, dt_relative_usage_score, zero_counter_score
    subset = customers_features_df[customers_features_df["ACCOUNT_NUMBER"].astype(str).isin(set(escalations_accounts))]
    if subset.empty:
        return (0.33, 0.33, 0.34), None, None

    # compute pre-optimization mean using session weights for pattern/relative/zero (or provided defaults)
    pre_wp = st.session_state.get("w_pattern", 0.2)
    pre_wr = st.session_state.get("w_relative", 0.2)
    pre_wz = st.session_state.get("w_zero", 0.05)
    pre_scores = (
        w_feeder * (1 - subset["feeder_billing_efficiency"]) +
        w_dt * (1 - subset["dt_billing_efficiency"]) +
        w_location * subset["location_trust_score"] +
        pre_wp * subset["pattern_deviation_score"] +
        pre_wr * subset["dt_relative_usage_score"] +
        pre_wz * subset["zero_counter_score"]
    )
    pre_mean = pre_scores.mean()

    best_mean = -np.inf
    best_combo = (pre_wp, pre_wr, pre_wz)

    grid = np.arange(0, 1 + 1e-9, step)
    for wp in grid:
        for wr in grid:
            for wz in grid:
                total = wp + wr + wz
                if total == 0:
                    continue
                wp_n, wr_n, wz_n = wp/total, wr/total, wz/total
                scores = (
                    w_feeder * (1 - subset["feeder_billing_efficiency"]) +
                    w_dt * (1 - subset["dt_billing_efficiency"]) +
                    w_location * subset["location_trust_score"] +
                    wp_n * subset["pattern_deviation_score"] +
                    wr_n * subset["dt_relative_usage_score"] +
                    wz_n * subset["zero_counter_score"]
                )
                mean_score = scores.mean()
                if mean_score > best_mean:
                    best_mean = mean_score
                    best_combo = (wp_n, wr_n, wz_n)
    return best_combo, pre_mean, best_mean

# ------------------------
# UI: Upload Excel
# ------------------------
st.subheader("Upload Excel File")
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.warning("Please upload an Excel file to proceed.")
    st.stop()

# Read sheets with converters (preserve strings)
try:
    sheets = pd.read_excel(
        uploaded_file,
        sheet_name=None,
        converters={
            "Feeder Data": {"Feeder": preserve_exact_string},
            "Transformer Data": {"New Unique DT Nomenclature": preserve_exact_string, "Ownership": preserve_exact_string, "Connection Status": preserve_exact_string},
            "Customer Data_PPM": {"NAME_OF_DT": preserve_exact_string, "NAME_OF_FEEDER": preserve_exact_string, "ACCOUNT_NUMBER": preserve_exact_string, "METER_NUMBER": preserve_exact_string, "BUSINESS_UNIT": preserve_exact_string, "UNDERTAKING": preserve_exact_string, "TARIFF": preserve_exact_string, "CUSTOMER_NAME": preserve_exact_string},
            "Customer Data_PPD": {"NAME_OF_DT": preserve_exact_string, "NAME_OF_FEEDER": preserve_exact_string, "ACCOUNT_NUMBER": preserve_exact_string, "METER_NUMBER": preserve_exact_string, "BUSINESS_UNIT": preserve_exact_string, "UNDERTAKING": preserve_exact_string, "TARIFF": preserve_exact_string, "CUSTOMER_NAME": preserve_exact_string},
            "Feeder Band": {"BAND": preserve_exact_string, "Feeder": preserve_exact_string, "Short Name": preserve_exact_string},
            "Customer Tariffs": {"Tariff": preserve_exact_string},
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

# keep raw copies for escalations lookup (before heavy normalization)
ppm_raw = ppm_df.copy() if ppm_df is not None else pd.DataFrame()
ppd_raw = ppd_df.copy() if ppd_df is not None else pd.DataFrame()
escalations_raw = escalations_df.copy() if escalations_df is not None else pd.DataFrame()

# validate presence
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
    missing_cols = [c for c in cols if df is None or c not in df.columns]
    if missing_cols:
        if name == "Transformer Data" and "Ownership" in missing_cols:
            st.warning("Ownership missing in Transformer Data - assuming PUBLIC")
            dt_df["Ownership"] = "PUBLIC"
        else:
            st.error(f"Missing columns in {name}: {missing_cols}")
            st.stop()

# ------------------------
# Months & pre-processing
# ------------------------
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
month_indices = {m: i for i, m in enumerate(months)}

# Fill missing month columns in necessary sheets and create (kWh) columns
for df, name in [(ppm_df, "Customer Data_PPM"), (ppd_df, "Customer Data_PPD"), (feeder_df, "Feeder Data"), (dt_df, "Transformer Data")]:
    missing_months = [m for m in months if m not in df.columns]
    if missing_months:
        for m in missing_months:
            df[m] = 0

# Handle missing simple columns
for df, col in [(ppm_df, "TARIFF"), (ppd_df, "TARIFF"), (band_df, "BAND"), (tariff_df, "Tariff")]:
    if col not in df.columns:
        df[col] = ""

# Short name for feeders if missing
if "Short Name" not in band_df.columns:
    band_df["Short Name"] = band_df["Feeder"].apply(lambda x: get_short_name(x))

# Rate handling
rate_col = next((c for c in ["Rate (NGN)", "Rate (₦)", "Rate", "RATE", "Rate(NGN)", "Rate(₦)"] if c in tariff_df.columns), None)
if rate_col:
    tariff_df["Rate (NGN)"] = pd.to_numeric(tariff_df[rate_col], errors="coerce").fillna(209.5)
else:
    tariff_df["Rate (NGN)"] = 209.5

# band rates
band_tariffs = {
    "A": ["A-MD1", "A-MD2", "A-Non MD"],
    "B": ["B-MD1", "B-MD2", "B-Non MD"],
    "C": ["C-MD1", "C-MD2", "C-Non MD"],
    "D": ["D-MD1", "D-MD2", "D-Non MD"],
    "E": ["E-MD1", "E-MD2", "E-Non MD"]
}
band_rates = {}
for band, tars in band_tariffs.items():
    rates = tariff_df[tariff_df["Tariff"].isin(tars)]["Rate (NGN)"]
    band_rates[band] = rates.mean() if not rates.empty else 209.5

# Merge feeder->band and tariff_rate
feeder_df = feeder_df.merge(band_df[["Feeder", "BAND"]], on="Feeder", how="left")
feeder_df["BAND"] = feeder_df["BAND"].fillna("Unknown")
feeder_df["Tariff_Rate"] = feeder_df["BAND"].map(band_rates).fillna(209.5)

# Expand monthly (kWh) columns (units: feeders were *1000 in original)
for month in months:
    # feeder_df months scaled by 1000
    feeder_df[f"{month} (kWh)"] = pd.to_numeric(feeder_df[month], errors="coerce").fillna(0) * 1000 if month in feeder_df.columns else 0
    # customers (ppm/ppd) months are assumed kWh already
    ppm_df[f"{month} (kWh)"] = pd.to_numeric(ppm_df[month], errors="coerce").fillna(0) if month in ppm_df.columns else 0
    ppd_df[f"{month} (kWh)"] = pd.to_numeric(ppd_df[month], errors="coerce").fillna(0) if month in ppd_df.columns else 0
    # dt
    dt_df[f"{month} (kWh)"] = pd.to_numeric(dt_df[month], errors="coerce").fillna(0) if month in dt_df.columns else 0

# drop original month columns to avoid confusion
for df in [feeder_df, dt_df, ppm_df, ppd_df]:
    df.drop(columns=[m for m in months if m in df.columns], errors='ignore', inplace=True)

# Filter NOT CONNECTED DTs with zero energy
dt_df["total_energy_kwh"] = dt_df[[f"{m} (kWh)" for m in months]].sum(axis=1)
not_connected_zero = (dt_df["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_df["total_energy_kwh"] == 0)
if not_connected_zero.any():
    st.warning(f"Excluding {not_connected_zero.sum()} DTs marked NOT CONNECTED with zero total energy.")
dt_df = dt_df[~not_connected_zero]

# Normalize name fields (note: account numbers left as-is)
normalization_pairs = [
    ("Feeder", feeder_df), ("NAME_OF_FEEDER", ppm_df), ("NAME_OF_FEEDER", ppd_df), ("Feeder", band_df),
    ("NAME_OF_DT", ppm_df), ("NAME_OF_DT", ppd_df), ("New Unique DT Nomenclature", dt_df),
    ("TARIFF", ppm_df), ("TARIFF", ppd_df), ("Tariff", tariff_df),
    ("BUSINESS_UNIT", ppm_df), ("BUSINESS_UNIT", ppd_df),
    ("UNDERTAKING", ppm_df), ("UNDERTAKING", ppd_df),
    ("Feeder", escalations_df), ("DT Nomenclature", escalations_df)
]
for col, df in normalization_pairs:
    if col in df.columns:
        df[col] = df[col].apply(normalize_name)

# Keep copies of combined customers before heavy filtering (full dataset to use for escalations lookup independent of UI filters)
ppm_df["Billing_Type"] = "PPM"
ppd_df["Billing_Type"] = "PPD"
customer_df_full = pd.concat([ppm_df, ppd_df], ignore_index=True, sort=False)

# Add Feeder column to full dataset (before filtering)
customer_df_full["Feeder"] = customer_df_full["NAME_OF_DT"].apply(
    lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x and len(x.split("-")) >= 3 else x
)
customer_df_full["Feeder"] = customer_df_full["Feeder"].apply(normalize_name)


if customer_df_full.empty:
    st.error("No customer rows found in PPM/PPD after load.")
    st.stop()

# For internal app processing we will continue with normalized + filtered dataset later
ppm_df_proc = ppm_df.copy()
ppd_df_proc = ppd_df.copy()
customer_df = pd.concat([ppm_df_proc, ppd_df_proc], ignore_index=True, sort=False)

# Validate DTs against transformer data (keep error report)
valid_dts = set(dt_df["New Unique DT Nomenclature"].unique())
customer_invalid_dts = customer_df[~customer_df["NAME_OF_DT"].isin(valid_dts)]
error_report = []
if not customer_invalid_dts.empty:
    for _, row in customer_invalid_dts.iterrows():
        error_report.append({
            "ACCOUNT_NUMBER": row.get("ACCOUNT_NUMBER"),
            "NAME_OF_DT": row.get("NAME_OF_DT"),
            "NAME_OF_FEEDER": row.get("NAME_OF_FEEDER"),
            "BUSINESS_UNIT": row.get("BUSINESS_UNIT"),
            "UNDERTAKING": row.get("UNDERTAKING"),
            "Reason": "NAME_OF_DT not in Transformer Data"
        })
error_report_df = pd.DataFrame(error_report)
# Filter out invalid dt customers for app flows (but their raw records remain in customer_df_full for escalations lookup)
customer_df = customer_df[customer_df["NAME_OF_DT"].isin(valid_dts)]
if customer_df.empty:
    st.error("No valid customers after filtering by Transformer Data.")
    st.stop()

# Create short names and feeder link columns
feeder_df["Feeder_Short"] = feeder_df["Feeder"].apply(lambda x: get_short_name(x))
dt_df["DT_Short_Name"] = dt_df["New Unique DT Nomenclature"].apply(lambda x: get_short_name(x, is_dt=True))
dt_df["Feeder"] = dt_df["New Unique DT Nomenclature"].apply(lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x and len(x.split("-")) >= 3 else x)
dt_df["Feeder"] = dt_df["Feeder"].apply(normalize_name)
dt_df["NAME_OF_DT"] = dt_df["New Unique DT Nomenclature"]
customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))
customer_df["Feeder"] = customer_df["NAME_OF_DT"].apply(lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x and len(x.split("-")) >= 3 else x)
customer_df["Feeder"] = customer_df["Feeder"].apply(normalize_name)

# Filter feeders
valid_feeders = set(feeder_df["Feeder"])
dt_df = dt_df[dt_df["Feeder"].isin(valid_feeders)]
customer_df = customer_df[customer_df["Feeder"].isin(valid_feeders)]
if dt_df.empty or customer_df.empty:
    st.error("No valid DTs or customers after filtering by Feeder Data.")
    st.stop()

# Merge tariff rates for customers
tariff_matches = customer_df["TARIFF"].isin(tariff_df["Tariff"])
if not tariff_matches.all():
    st.warning(f"Some tariffs in customer data not found in tariffs list: {customer_df[~tariff_matches]['TARIFF'].unique()}")
customer_df = customer_df.merge(tariff_df[["Tariff", "Rate (NGN)"]], left_on="TARIFF", right_on="Tariff", how="left")
customer_df["Rate (NGN)"] = customer_df["Rate (NGN)"].fillna(209.5)
customer_df.drop(columns=["Tariff"], errors="ignore", inplace=True)

# Map dt to feeder tariff rates
dt_df = dt_df.merge(feeder_df[["Feeder", "Tariff_Rate"]], on="Feeder", how="left")
dt_df["Tariff_Rate"] = dt_df["Tariff_Rate"].fillna(209.5)

# Location trust scores from escalations (using normalized feeder/dt names)
escalations_df["Report_Count"] = 1
feeder_escalations = escalations_df.groupby("Feeder")["Report_Count"].sum().reset_index()
if not feeder_escalations.empty:
    feeder_escalations["location_trust_score"] = feeder_escalations["Report_Count"] / feeder_escalations["Report_Count"].max()
else:
    feeder_escalations["location_trust_score"] = 0
feeder_escalations["location_trust_score"] = feeder_escalations["location_trust_score"].fillna(0).clip(0, 1)

dt_escalations = escalations_df.groupby("DT Nomenclature")["Report_Count"].sum().reset_index()
if not dt_escalations.empty:
    dt_escalations["location_trust_score"] = dt_escalations["Report_Count"] / dt_escalations["Report_Count"].max()
else:
    dt_escalations["location_trust_score"] = 0
dt_escalations["location_trust_score"] = dt_escalations["location_trust_score"].fillna(0).clip(0, 1)

# ------------------------
# Create monthly (melted) data - full (for escalations report) and filtered (app)
# ------------------------
# required id vars for melt
required_id_vars = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type", "Feeder", "Rate (NGN)"]
value_vars = [f"{m} (kWh)" for m in months]
missing_id_vars = [v for v in required_id_vars if v not in customer_df.columns]
if missing_id_vars:
    st.error(f"Missing id_vars in customer_df for melt: {missing_id_vars}")
    st.stop()

# Melt full dataset (independent of UI filters) for escalations reporting
customer_monthly_all = customer_df_full.melt(
    id_vars=[c for c in required_id_vars if c in customer_df_full.columns],
    value_vars=[v for v in value_vars if v in customer_df_full.columns],
    var_name="month", value_name="billed_kwh"
) if not customer_df_full.empty else pd.DataFrame()

# Standardize month strings
if not customer_monthly_all.empty:
    customer_monthly_all["month"] = customer_monthly_all["month"].str.replace(" (kWh)", "", regex=False)
    customer_monthly_all["month"] = pd.Categorical(customer_monthly_all["month"], categories=months, ordered=True)
    # ensure numeric
    customer_monthly_all["billed_kwh"] = pd.to_numeric(customer_monthly_all["billed_kwh"], errors="coerce").fillna(0)

# For the main app flows we will show filtered months later (after user selection)
# Melt processed (post-filter) customer data to customer_monthly (we'll create after UI month selection)

# DT melt all
dt_agg_monthly_all = dt_df.melt(
    id_vars=["New Unique DT Nomenclature", "DT_Short_Name", "Feeder", "Tariff_Rate", "Ownership", "Connection Status", "total_energy_kwh"],
    value_vars=[f"{m} (kWh)" for m in months],
    var_name="month", value_name="total_dt_kwh"
)
dt_agg_monthly_all["month"] = dt_agg_monthly_all["month"].str.replace(" (kWh)", "", regex=False)
dt_agg_monthly_all["month"] = pd.Categorical(dt_agg_monthly_all["month"], categories=months, ordered=True)
dt_agg_monthly_all["total_dt_kwh"] = pd.to_numeric(dt_agg_monthly_all["total_dt_kwh"], errors="coerce").fillna(0)

# Customer-month aggregation for full data
if not customer_monthly_all.empty:
    cust_agg_all = customer_monthly_all.groupby(["NAME_OF_DT", "Feeder"])["billed_kwh"].sum().reset_index().rename(columns={"billed_kwh": "customer_billed_kwh"})
else:
    cust_agg_all = pd.DataFrame(columns=["NAME_OF_DT", "Feeder", "customer_billed_kwh"])

# DT aggregate sums (all months)
dt_agg_sum_all = dt_agg_monthly_all.groupby(["New Unique DT Nomenclature", "DT_Short_Name", "Feeder", "Tariff_Rate", "Ownership", "Connection Status", "total_energy_kwh"], dropna=False)["total_dt_kwh"].sum().reset_index()

# DT billing efficiency (all months)
dt_merged_all = dt_agg_sum_all.merge(cust_agg_all, left_on=["New Unique DT Nomenclature", "Feeder"], right_on=["NAME_OF_DT", "Feeder"], how="left")
dt_merged_all["customer_billed_kwh"] = dt_merged_all["customer_billed_kwh"].fillna(0)
dt_merged_all["total_billed_kwh"] = np.where(
    dt_merged_all["Ownership"].str.strip().str.upper().isin(["PRIVATE"]),
    dt_merged_all["total_dt_kwh"],
    dt_merged_all["customer_billed_kwh"]
)
dt_merged_all["dt_billing_efficiency"] = np.where(
    (dt_merged_all["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_merged_all["total_energy_kwh"] > 0),
    0.0,
    (dt_merged_all["total_billed_kwh"] / dt_merged_all["total_dt_kwh"].replace(0, 1)).clip(0, 1)
)
dt_merged_all["energy_lost_kwh"] = dt_merged_all["total_dt_kwh"] - dt_merged_all["total_billed_kwh"]
dt_merged_all["financial_loss_naira"] = dt_merged_all["energy_lost_kwh"] * dt_merged_all["Tariff_Rate"]

# Feeder aggregation (all months)
feeder_monthly_all = feeder_df.melt(id_vars=["Feeder", "Feeder_Short", "Tariff_Rate"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="feeder_energy_kwh")
feeder_monthly_all["month"] = feeder_monthly_all["month"].str.replace(" (kWh)", "", regex=False)
feeder_monthly_all["month"] = pd.Categorical(feeder_monthly_all["month"], categories=months, ordered=True)
feeder_monthly_all["feeder_energy_kwh"] = pd.to_numeric(feeder_monthly_all["feeder_energy_kwh"], errors="coerce").fillna(0)
feeder_agg_all = feeder_monthly_all.groupby(["Feeder", "Feeder_Short", "Tariff_Rate"])["feeder_energy_kwh"].sum().reset_index()

feeder_agg_billed_all = dt_merged_all.groupby(["Feeder"])["total_billed_kwh"].sum().reset_index()
feeder_merged_all = feeder_agg_all.merge(feeder_agg_billed_all, on=["Feeder"], how="left")
feeder_merged_all["total_billed_kwh"] = feeder_merged_all["total_billed_kwh"].fillna(0)
feeder_merged_all["feeder_billing_efficiency"] = (feeder_merged_all["total_billed_kwh"] / feeder_merged_all["feeder_energy_kwh"].replace(0, 1)).clip(0, 1)
feeder_merged_all["feeder_energy_lost_kwh"] = feeder_merged_all["feeder_energy_kwh"] - feeder_merged_all["total_billed_kwh"]
feeder_merged_all["feeder_financial_loss_naira"] = feeder_merged_all["feeder_energy_lost_kwh"] * feeder_merged_all["Tariff_Rate"]

# Attach location trust scores to feeder_merged_all and dt_merged_all
feeder_merged_all = feeder_merged_all.merge(feeder_escalations[["Feeder", "location_trust_score"]], on="Feeder", how="left")
feeder_merged_all["location_trust_score"] = feeder_merged_all["location_trust_score"].fillna(0)
dt_merged_all = dt_merged_all.merge(dt_escalations[["DT Nomenclature", "location_trust_score"]], left_on="New Unique DT Nomenclature", right_on="DT Nomenclature", how="left")
dt_merged_all["location_trust_score"] = dt_merged_all["location_trust_score"].fillna(0)

# ------------------------
# Precompute per-customer aggregated features (for escalations report)
# ------------------------
# Unique customer rows from processed combined customer_df (these survived filtering)
customers_base = customer_df[["ACCOUNT_NUMBER", "CUSTOMER_NAME", "ADDRESS", " Feeder" ]] if " Feeder" in customer_df.columns else customer_df.copy()
# We'll build customers_features across ACCOUNT_NUMBER using processed customer_df (survived DT/Feeder filtering)
customers_features = customer_df[["ACCOUNT_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Feeder", "NAME_OF_DT", "METER_NUMBER", "Billing_Type", "Rate (NGN)"]].drop_duplicates(subset=["ACCOUNT_NUMBER"]).copy()
customers_features.rename(columns={"NAME_OF_DT": "NAME_OF_DT"}, inplace=True)

# Add feeder_billing_efficiency by mapping from feeder_merged_all
customers_features = customers_features.merge(feeder_merged_all[["Feeder", "feeder_billing_efficiency", "location_trust_score"]], on="Feeder", how="left")
customers_features["feeder_billing_efficiency"] = customers_features["feeder_billing_efficiency"].fillna(0)
# Add DT billing efficiency by mapping
dt_eff_map = dt_merged_all[["New Unique DT Nomenclature", "dt_billing_efficiency", "location_trust_score"]].groupby("New Unique DT Nomenclature").agg({
    "dt_billing_efficiency": "mean",
    "location_trust_score": "mean"
}).reset_index()
customers_features = customers_features.merge(dt_eff_map, left_on="NAME_OF_DT", right_on="New Unique DT Nomenclature", how="left")
customers_features["dt_billing_efficiency"] = customers_features["dt_billing_efficiency"].fillna(0)
# Combine location trust (prefer DT-level then feeder level)
customers_features["location_trust_score"] = customers_features["location_trust_score_x"].combine_first(customers_features["location_trust_score_y"]).fillna(0)
customers_features.drop(columns=["location_trust_score_x", "location_trust_score_y", "New Unique DT Nomenclature"], errors="ignore", inplace=True)

# Pattern & zero counters computed on wide customer_df (processed wide form)
pattern_all = calculate_pattern_deviation(customer_df, "ACCOUNT_NUMBER", value_vars)
zero_all = calculate_zero_counter(customer_df, "ACCOUNT_NUMBER", value_vars)
customers_features = customers_features.merge(pattern_all, left_on="ACCOUNT_NUMBER", right_on="id", how="left")
customers_features = customers_features.merge(zero_all, left_on="ACCOUNT_NUMBER", right_on="id", how="left")
customers_features[["pattern_deviation_score", "zero_counter_score"]] = customers_features[["pattern_deviation_score", "zero_counter_score"]].fillna(0)

# DT relative usage computed from customer_monthly_all
dt_rel_all = calculate_dt_relative_usage(customer_monthly_all) if not customer_monthly_all.empty else pd.DataFrame(columns=["ACCOUNT_NUMBER", "dt_relative_usage_score"])
customers_features = customers_features.merge(dt_rel_all, on="ACCOUNT_NUMBER", how="left")
customers_features["dt_relative_usage_score"] = customers_features["dt_relative_usage_score"].fillna(0)

# ------------------------
# Streamlit UI: Filters (these apply to main app visualizations, *not* the escalations report)
# ------------------------
st.title("Ikeja Electric Energy Theft Detection Dashboard")
st.subheader("Filters")
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    bu_options = sorted(customer_df["BUSINESS_UNIT"].unique())
    if bu_options:
        selected_bu = st.selectbox("Select Business Unit", bu_options)
    else:
        selected_bu = None
        st.warning("No Business Units available.")
with col2:
    if selected_bu:
        customer_df_bu = customer_df[customer_df["BUSINESS_UNIT"] == selected_bu]
        ut_options = sorted(customer_df_bu["UNDERTAKING"].unique())
        if ut_options:
            selected_ut = st.selectbox("Select Undertaking", ut_options)
        else:
            selected_ut = None
            st.warning("No Undertakings available for selected BU.")
    else:
        customer_df_bu = customer_df.copy()
        selected_ut = None
with col3:
    feeder_options = sorted(feeder_df["Feeder_Short"].unique())
    if feeder_options:
        selected_feeder_short = st.selectbox("Select Feeder", feeder_options)
    else:
        selected_feeder_short = None
with col4:
    if selected_feeder_short:
        selected_feeder = feeder_df[feeder_df["Feeder_Short"] == selected_feeder_short]["Feeder"].iloc[0]
        dt_df_filtered = dt_df[dt_df["Feeder"] == selected_feeder]
        dt_options = sorted(dt_df_filtered["DT_Short_Name"].unique())
        if dt_options:
            selected_dt_short = st.selectbox("Select DT", dt_options)
        else:
            selected_dt_short = None
    else:
        selected_feeder = None
        selected_dt_short = None
with col5:
    start_month = st.selectbox("Start Month", months)
with col6:
    end_month = st.selectbox("End Month", months, index=len(months)-1)
    if month_indices[start_month] > month_indices[end_month]:
        st.error("Start Month must be before or equal to End Month.")
        st.stop()

# ------------------------
# Weight sliders + optimizer controls
# ------------------------
st.subheader("Adjust Theft Probability Weights")
if "w_pattern" not in st.session_state:
    st.session_state.w_pattern = 0.2
if "w_relative" not in st.session_state:
    st.session_state.w_relative = 0.2
if "w_zero" not in st.session_state:
    st.session_state.w_zero = 0.05

colw1, colw2, colw3 = st.columns(3)
with colw1:
    w_feeder = st.slider("Feeder Billing Efficiency Weight", 0.0, 1.0, 0.2, 0.01)
    w_dt = st.slider("DT Billing Efficiency Weight", 0.0, 1.0, 0.2, 0.01)
with colw2:
    w_location = st.slider("Location Trust Score Weight", 0.0, 1.0, 0.2, 0.01)
    w_pattern = st.slider("Consumption Pattern Deviation Weight", 0.0, 1.0, st.session_state.w_pattern, 0.01, key="w_pattern")
with colw3:
    w_relative = st.slider("DT Relative Usage Score Weight", 0.0, 1.0, st.session_state.w_relative, 0.01, key="w_relative")
    w_zero = st.slider("Zero Frequency Weight", 0.0, 1.0, st.session_state.w_zero, 0.01, key="w_zero")

# Optimize button (this operates on the precomputed customers_features and escalations accounts)
if st.button("Optimize Customer-Level Weights for Escalations"):
    escal_accs = escalations_df["Account No"].astype(str).str.strip().tolist()
    (best_wp, best_wr, best_wz), pre_mean, post_mean = optimize_customer_weights_for_escalations(
        customers_features[["ACCOUNT_NUMBER", "feeder_billing_efficiency", "dt_billing_efficiency", "location_trust_score", "pattern_deviation_score", "dt_relative_usage_score", "zero_counter_score"]],
        escal_accs,
        w_feeder, w_dt, w_location,
        step=0.05
    )
    # Update session sliders
    st.session_state.w_pattern = float(best_wp)
    st.session_state.w_relative = float(best_wr)
    st.session_state.w_zero = float(best_wz)
    st.success("Optimization finished. Pattern/Relative/Zero sliders updated.")
    if pre_mean is not None:
        st.info(f"Average theft probability for escalation customers improved from {pre_mean:.3f} → {post_mean:.3f}")
    st.experimental_rerun()

# Normalize weights
weights_sum = w_feeder + w_dt + w_location + w_pattern + w_relative + w_zero
if weights_sum == 0:
    st.error("Total weight cannot be zero.")
    st.stop()
w_feeder /= weights_sum
w_dt /= weights_sum
w_location /= weights_sum
w_pattern /= weights_sum
w_relative /= weights_sum
w_zero /= weights_sum

# ------------------------
# Now create the filtered customer_monthly (for main app visuals) based on selected months
# ------------------------
# melt the processed customer_df (we earlier created customer_df as processed) and then filter months range
customer_monthly = customer_df.melt(
    id_vars=required_id_vars,
    value_vars=value_vars,
    var_name="month",
    value_name="billed_kwh"
)
customer_monthly["month"] = customer_monthly["month"].str.replace(" (kWh)", "", regex=False)
customer_monthly["month"] = pd.Categorical(customer_monthly["month"], categories=months, ordered=True)
customer_monthly["billed_kwh"] = pd.to_numeric(customer_monthly["billed_kwh"], errors="coerce").fillna(0)

selected_months = months[month_indices[start_month]: month_indices[end_month] + 1]
customer_monthly = customer_monthly[customer_monthly["month"].isin(selected_months)]
dt_agg_monthly = dt_agg_monthly_all[dt_agg_monthly_all["month"].isin(selected_months)]
feeder_monthly = feeder_monthly_all[feeder_monthly_all["month"].isin(selected_months)]

if customer_monthly.empty:
    st.warning("No customer monthly data for the selected month range. Visualizations will be limited.")

# ------------------------
# Recalculate dt_relative_usage for the selected months (for app's main flows)
# ------------------------
dt_relative_usage = calculate_dt_relative_usage(customer_monthly)

# calculate pattern and zero counters on processed wide customer_df (we already computed pattern_all, zero_all earlier)
pattern_df = calculate_pattern_deviation(customer_df, "ACCOUNT_NUMBER", value_vars)
zero_df = calculate_zero_counter(customer_df, "ACCOUNT_NUMBER", value_vars)

# Merge relevant scores to customer_monthly for per-month scoring
customer_monthly = customer_monthly.merge(feeder_merged_all[["Feeder", "feeder_billing_efficiency", "location_trust_score"]], on="Feeder", how="left")
customer_monthly = customer_monthly.merge(dt_merged_all[["New Unique DT Nomenclature", "dt_billing_efficiency", "location_trust_score"]], left_on=["NAME_OF_DT"], right_on=["New Unique DT Nomenclature"], how="left", suffixes=("_feeder", "_dt"))
customer_monthly = customer_monthly.merge(pattern_df, left_on=["ACCOUNT_NUMBER"], right_on=["id"], how="left")
customer_monthly = customer_monthly.merge(zero_df, left_on=["ACCOUNT_NUMBER"], right_on=["id"], how="left")
customer_monthly = customer_monthly.merge(dt_relative_usage, on="ACCOUNT_NUMBER", how="left")

# fillna
for col in ["feeder_billing_efficiency", "dt_billing_efficiency", "location_trust_score", "pattern_deviation_score", "zero_counter_score", "dt_relative_usage_score"]:
    if col in customer_monthly.columns:
        customer_monthly[col] = customer_monthly[col].fillna(0)
    else:
        customer_monthly[col] = 0

# compute per-month theft probability
customer_monthly["theft_probability"] = (
    w_feeder * (1 - customer_monthly["feeder_billing_efficiency"]) +
    w_dt * (1 - customer_monthly["dt_billing_efficiency"]) +
    w_location * customer_monthly["location_trust_score"] +
    w_pattern * customer_monthly["pattern_deviation_score"] +
    w_relative * customer_monthly["dt_relative_usage_score"] +
    w_zero * customer_monthly["zero_counter_score"]
).clip(0, 1)

# Aggregate per-account across selected months for the app display & downloads
month_customers = customer_monthly.groupby(["ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type"]).agg({
    "billed_kwh": "sum",
    "theft_probability": "mean",
    "pattern_deviation_score": "mean",
    "dt_relative_usage_score": "mean",
    "zero_counter_score": "mean"
}).reset_index()
month_customers = month_customers.sort_values("theft_probability", ascending=False)

# Add risk tier
month_customers["risk_tier"] = pd.cut(month_customers["theft_probability"], bins=[0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"], include_lowest=True)

# ------------------------
# Escalations Report (independent of filters)
# ------------------------
def generate_escalations_report(prepaid_raw_df, postpaid_raw_df, escalations_raw_df, customers_features_df, theft_scores_map):
    """
    Build escalations report for all accounts in escalations_raw_df['Account No'].
    prepaid_raw_df / postpaid_raw_df: raw data (before app filters)
    customers_features_df: per-account aggregated features (from processed dataset)
    theft_scores_map: dict mapping ACCOUNT_NUMBER (str) -> theft_probability (float) computed on full data (aggregated)
    """
    accounts = escalations_raw_df["Account No"].astype(str).str.strip().tolist()
    results = []
    # combine raw lists for lookup (keep both sources)
    prepaid_lookup = prepaid_raw_df.set_index(prepaid_raw_df["ACCOUNT_NUMBER"].astype(str)) if not prepaid_raw_df.empty else pd.DataFrame()
    postpaid_lookup = postpaid_raw_df.set_index(postpaid_raw_df["ACCOUNT_NUMBER"].astype(str)) if not postpaid_raw_df.empty else pd.DataFrame()

    for acc in accounts:
        acc_s = str(acc).strip()
        found_in = []
        row_data = {"Account No": acc_s}
        # Try prepaid
        rec_ppm = None
        rec_ppd = None
        if not prepaid_raw_df.empty and acc_s in prepaid_lookup.index:
            # if multiple matches take first
            rec_ppm = prepaid_lookup.loc[acc_s]
            if isinstance(rec_ppm, pd.DataFrame):
                rec_ppm = rec_ppm.iloc[0]
            found_in.append("PPM")
        if not postpaid_raw_df.empty and acc_s in postpaid_lookup.index:
            rec_ppd = postpaid_lookup.loc[acc_s]
            if isinstance(rec_ppd, pd.DataFrame):
                rec_ppd = rec_ppd.iloc[0]
            found_in.append("PPD")

        if not found_in:
            row_data.update({
                "Found In": "Not Found",
                "Customer Name": "",
                "Feeder": "",
                "DT": "",
                **{m: np.nan for m in months},
                "Theft Probability": np.nan,
                "Risk Tier": "",
                "Status": "Not Found"
            })
            results.append(row_data)
            continue

        # Prefer PPD record if exists, else PPM (but include both indicators)
        source_rec = rec_ppd if rec_ppd is not None else rec_ppm
        # Extract fields safely
        cust_name = source_rec.get("CUSTOMER_NAME", "") if hasattr(source_rec, "get") else (source_rec["CUSTOMER_NAME"] if "CUSTOMER_NAME" in source_rec.index else "")
        feeder_name = source_rec.get("NAME_OF_FEEDER", "") if hasattr(source_rec, "get") else (source_rec["NAME_OF_FEEDER"] if "NAME_OF_FEEDER" in source_rec.index else "")
        dt_name = source_rec.get("NAME_OF_DT", "") if hasattr(source_rec, "get") else (source_rec["NAME_OF_DT"] if "NAME_OF_DT" in source_rec.index else "")

        # monthly readings from raw row (JAN..JUN)
        month_readings = {}
        for m in months:
            if m in source_rec.index:
                month_readings[m] = source_rec[m]
            else:
                # fallback to (kWh) column if raw had that
                kcol = f"{m} (kWh)"
                month_readings[m] = source_rec[kcol] if kcol in source_rec.index else np.nan

        # theft probability mapping (from processed aggregated customers_features or theft_scores_map)
        tp = theft_scores_map.get(acc_s)
        risk = ""
        if tp is not None and not np.isnan(tp):
            if tp <= 0.4:
                risk = "Low"
            elif tp <= 0.7:
                risk = "Medium"
            else:
                risk = "High"

        row_data.update({
            "Found In": ",".join(found_in),
            "Customer Name": cust_name,
            "Feeder": feeder_name,
            "DT": dt_name,
            **{m: month_readings[m] for m in months},
            "Theft Probability": tp if tp is not None else np.nan,
            "Risk Tier": risk,
            "Status": "Found"
        })
        results.append(row_data)
    return pd.DataFrame(results)

# Build theft_scores_map for full customers_features (aggregated per-account)
# customers_features has per-account features (feeder_billing_efficiency, dt_billing_efficiency, location_trust_score, pattern_deviation_score, dt_relative_usage_score, zero_counter_score)
customers_features_for_score = customers_features.copy()
# ensure ACCOUNT_NUMBER is string
customers_features_for_score["ACCOUNT_NUMBER"] = customers_features_for_score["ACCOUNT_NUMBER"].astype(str)
customers_features_for_score["computed_theft_prob"] = (
    w_feeder * (1 - customers_features_for_score["feeder_billing_efficiency"]) +
    w_dt * (1 - customers_features_for_score["dt_billing_efficiency"]) +
    w_location * customers_features_for_score["location_trust_score"] +
    w_pattern * customers_features_for_score["pattern_deviation_score"] +
    w_relative * customers_features_for_score["dt_relative_usage_score"] +
    w_zero * customers_features_for_score["zero_counter_score"]
).clip(0, 1)

theft_scores_map = pd.Series(customers_features_for_score["computed_theft_prob"].values, index=customers_features_for_score["ACCOUNT_NUMBER"].astype(str)).to_dict()

# Escalations Report UI
st.subheader("Escalations Report (checks all Account No in Escalations sheet)")
if st.button("Generate Escalations Report"):
    report_df = generate_escalations_report(ppm_raw, ppd_raw, escalations_raw, customers_features_for_score, theft_scores_map)
    st.success(f"Escalations report generated: {len(report_df)} rows")
    st.dataframe(report_df, use_container_width=True)

    # prepare excel download
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        report_df.to_excel(writer, index=False, sheet_name="Escalations Report")
        # also include a sheet with customers_features for reference
        customers_features_for_score.to_excel(writer, index=False, sheet_name="Customers Features")
    buffer.seek(0)
    st.download_button(
        label="📥 Download Escalations Report (Excel)",
        data=buffer,
        file_name=f"Escalations_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ------------------------
# Main app visualizations & download (existing flows)
# ------------------------
st.subheader(f"Customers under {selected_dt_short or 'Selected DT'} ({selected_feeder_short or 'Selected Feeder'}, {start_month} to {end_month})")
if selected_dt_short:
    filtered_customers = customer_monthly[customer_monthly["DT_Short_Name"] == selected_dt_short]
    if filtered_customers.empty:
        st.warning("No customers for this DT & selected months.")
    else:
        # Heatmap: top N (user chooses)
        customer_means = filtered_customers.groupby("ACCOUNT_NUMBER")["theft_probability"].mean().reset_index()
        customer_order = customer_means.sort_values("theft_probability", ascending=False)["ACCOUNT_NUMBER"].tolist()
        num_customers = st.number_input("Number of high-risk customers for Heatmap (0 for all)", min_value=0, value=min(10, len(customer_order)), step=1)
        if num_customers > 0:
            heat_customers = customer_order[:num_customers]
            filtered_for_heat = filtered_customers[filtered_customers["ACCOUNT_NUMBER"].isin(heat_customers)]
        else:
            filtered_for_heat = filtered_customers
        pivot_heat = filtered_for_heat.pivot_table(index="ACCOUNT_NUMBER", columns="month", values="theft_probability", aggfunc="mean").reindex(index=customer_order[:num_customers or None], columns=months)
        if not pivot_heat.empty:
            plt.figure(figsize=(10, max(4, 0.5 * pivot_heat.shape[0])))
            sns.heatmap(pivot_heat, cmap="YlOrRd", vmin=0, vmax=1, cbar_kws={"label": "Theft Probability"})
            plt.title(f"Theft Probability for {selected_dt_short} ({period_label if 'period_label' in locals() else ''})")
            st.pyplot(plt.gcf())
            plt.close()

# Customer list & export (from month_customers)
st.subheader("Customer List (aggregated across selected months)")
if not month_customers.empty:
    display_cols = ["ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "billed_kwh", "Billing_Type", "theft_probability", "risk_tier", "pattern_deviation_score", "dt_relative_usage_score", "zero_counter_score"]
    present_cols = [c for c in display_cols if c in month_customers.columns]
    st.dataframe(month_customers[present_cols].sort_values("theft_probability", ascending=False), use_container_width=True)
    csv = month_customers.to_csv(index=False)
    st.download_button(label=f"Download Customer List ({start_month} to {end_month})", data=csv, file_name=f"theft_analysis_{start_month}_{end_month}.csv", mime="text/csv")
else:
    st.info("No aggregated customer data (generate a DT/feeder and month range first).")

# Footer
st.markdown("Built by Elvis Ebenuwah for Ikeja Electric. 2025.")
