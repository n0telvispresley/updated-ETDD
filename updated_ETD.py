import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Streamlit config
st.set_page_config(page_title="IE Energy Theft Detection Dashboard", layout="wide")

# ------------------------
# Utility functions
# ------------------------
def preserve_exact_string(value):
    if pd.isna(value) or value is None:
        return ""
    return str(value)

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
# Pattern deviation: improved
# - Flags months less than 60% of customer's max month
# - If max == 0 (all zeros) => score 1.0
# ------------------------
def calculate_pattern_deviation(df, id_col, value_cols):
    results = []
    valid_value_cols = [c for c in value_cols if c in df.columns]
    for id_val, group in df.groupby(id_col):
        try:
            values = group[valid_value_cols].iloc[0].values.astype(float)
        except Exception:
            values = np.zeros(len(valid_value_cols))
        max_val = np.max(values) if len(values) > 0 else 0.0
        if max_val == 0:
            score = 1.0
        else:
            below = np.sum(values < 0.6 * max_val)
            score = below / len(valid_value_cols)
        results.append({"id": id_val, "pattern_deviation_score": min(score, 1.0)})
    return pd.DataFrame(results)

# ------------------------
# Zero counter (zero frequency)
# - score = fraction of months that are zero
# ------------------------
def calculate_zero_counter(df, id_col, value_cols):
    results = []
    valid_value_cols = [c for c in value_cols if c in df.columns]
    for id_val, group in df.groupby(id_col):
        try:
            values = group[valid_value_cols].iloc[0].values.astype(float)
        except Exception:
            values = np.zeros(len(valid_value_cols))
        zeros = np.sum(values == 0)
        score = zeros / len(valid_value_cols) if len(valid_value_cols) > 0 else 0.0
        results.append({"id": id_val, "zero_counter_score": min(score, 1.0)})
    return pd.DataFrame(results)

# ------------------------
# DT relative usage score (existing algorithm)
# ------------------------
def calculate_dt_relative_usage(customer_monthly, selected_months):
    customer_agg = customer_monthly.groupby(["ACCOUNT_NUMBER", "NAME_OF_DT"])["billed_kwh"].sum().reset_index()
    dt_avg = customer_agg.groupby("NAME_OF_DT")["billed_kwh"].mean().reset_index().rename(columns={"billed_kwh": "dt_avg_kwh"})
    customer_agg = customer_agg.merge(dt_avg, on="NAME_OF_DT", how="left")
    customer_agg["relative_ratio"] = np.where(customer_agg["dt_avg_kwh"] == 0, 0.5, customer_agg["billed_kwh"] / customer_agg["dt_avg_kwh"])
    customer_agg["dt_relative_usage_score"] = customer_agg.apply(
        lambda row: 0.9 if row["billed_kwh"] < row["dt_avg_kwh"] * 0.3
        else 0.1 if row["billed_kwh"] > row["dt_avg_kwh"] * 0.7
        else 0.1 + (0.9 - 0.1) * (0.7 - row["relative_ratio"]) / (0.7 - 0.3),
        axis=1
    ).clip(0, 1)
    return customer_agg[["ACCOUNT_NUMBER", "dt_relative_usage_score"]]

# ------------------------
# Weight optimizer for customer-level sub-weights:
# - grid search with step 0.05 over (pattern, relative, zero)
# - keeps the other global weights (feeder, dt, location) fixed
# - objective: maximize average theft_probability across escalations customers
# ------------------------
def optimize_customer_weights(customer_df, escalations_accounts, w_feeder, w_dt, w_location):
    # customer_df must already have the following columns:
    # ACCOUNT_NUMBER, feeder_billing_efficiency, dt_billing_efficiency,
    # location_trust_score, pattern_deviation_score, dt_relative_usage_score, zero_counter_score
    subset = customer_df[customer_df["ACCOUNT_NUMBER"].astype(str).isin(escalations_accounts)].copy()
    if subset.empty:
        return (0.33, 0.33, 0.34)  # fallback equal split
    best_combo = (0.33, 0.33, 0.34)
    best_mean = -np.inf

    # grid step 0.05
    grid = np.arange(0.0, 1.0001, 0.05)
    for wp in grid:
        for wr in grid:
            for wz in grid:
                s = wp + wr + wz
                if s == 0:
                    continue
                wp_n, wr_n, wz_n = wp / s, wr / s, wz / s
                theft_scores = (
                    w_feeder * (1 - subset["feeder_billing_efficiency"]) +
                    w_dt * (1 - subset["dt_billing_efficiency"]) +
                    w_location * subset["location_trust_score"] +
                    wp_n * subset["pattern_deviation_score"] +
                    wr_n * subset["dt_relative_usage_score"] +
                    wz_n * subset["zero_counter_score"]
                )
                mean_score = theft_scores.mean()
                if mean_score > best_mean:
                    best_mean = mean_score
                    best_combo = (wp_n, wr_n, wz_n)
    return best_combo

# ------------------------
# Escalations report generator
# - takes prepaid and postpaid raw (unfiltered) sheets (ppm and ppd)
# - cross-references Escalations' "Account No"
# - outputs per-account rows (if multiple months -- contains monthly readings + theft_score)
# ------------------------
def generate_escalations_report(prepaid_df, postpaid_df, escalations_df, customer_monthly_df):
    # Make list of account numbers from Escalations (cast to str)
    escal_accounts = escalations_df["Account No"].astype(str).str.strip().unique().tolist()

    # Combine prepaid and postpaid (raw customer rows)
    customers_long = pd.concat([prepaid_df, postpaid_df], ignore_index=True)
    customers_long["ACCOUNT_NUMBER"] = customers_long["ACCOUNT_NUMBER"].astype(str).str.strip()

    # Prepare lookup from customer_monthly_df to get theft_probability and monthly readings
    # customer_monthly_df has ACCOUNT_NUMBER, month, billed_kwh, theft_probability
    # We'll pivot monthly billed_kwh to columns for the report
    monthly_pivot = customer_monthly_df.pivot_table(
        index=["ACCOUNT_NUMBER", "CUSTOMER_NAME", "ADDRESS", "NAME_OF_DT", "DT_Short_Name", "Feeder", "Billing_Type"],
        columns="month",
        values="billed_kwh",
        aggfunc="sum"
    ).reset_index()
    # Bring theft_probability aggregated across selected months (mean)
    theft_by_acc = customer_monthly_df.groupby("ACCOUNT_NUMBER")["theft_probability"].mean().reset_index().rename(columns={"theft_probability":"theft_probability_mean"})

    report_rows = []
    for acc in escal_accounts:
        matched_rows = customers_long[customers_long["ACCOUNT_NUMBER"] == acc]
        if matched_rows.empty:
            # Not found
            report_rows.append({
                "Account No": acc,
                "Found": False,
                "Customer Name": None,
                "Feeder": None,
                "DT": None,
                "Billing_Type": None,
                **{m: None for m in customer_monthly_df["month"].cat.categories} if "month" in customer_monthly_df.columns else {}
            })
        else:
            # There may be multiple matches (unlikely) - include one row per match
            for _, mr in matched_rows.iterrows():
                acc_str = str(mr.get("ACCOUNT_NUMBER", "")).strip()
                # get pivoted monthly readings if available
                pivot_row = monthly_pivot[monthly_pivot["ACCOUNT_NUMBER"] == acc_str]
                monthly_values = {}
                if not pivot_row.empty:
                    # pivot_row has columns: ACCOUNT_NUMBER, CUSTOMER_NAME, ADDRESS, NAME_OF_DT, DT_Short_Name, Feeder, Billing_Type, <months...>
                    for col in pivot_row.columns:
                        if col in ["ACCOUNT_NUMBER", "CUSTOMER_NAME", "ADDRESS", "NAME_OF_DT", "DT_Short_Name", "Feeder", "Billing_Type"]:
                            continue
                        monthly_values[col] = pivot_row.iloc[0][col]
                else:
                    # no monthly data present (maybe filtered out earlier); fill months with NaN
                    for mon in customer_monthly_df["month"].cat.categories:
                        monthly_values[mon] = np.nan

                theft_row = theft_by_acc[theft_by_acc["ACCOUNT_NUMBER"] == acc_str]
                theft_score = theft_row["theft_probability_mean"].iloc[0] if not theft_row.empty else np.nan

                report_rows.append({
                    "Account No": acc_str,
                    "Found": True,
                    "Customer Name": mr.get("CUSTOMER_NAME", ""),
                    "Feeder": mr.get("FEEDER", ""),
                    "DT": mr.get("NAME_OF_DT", ""),
                    "Billing_Type": mr.get("Billing_Type", ""),
                    "Theft Score (weighted)": theft_score,
                    **monthly_values
                })

    report_df = pd.DataFrame(report_rows)
    return report_df

# ------------------------
# START: Load Excel and original pipeline (kept close to your original code)
# ------------------------
st.subheader("Upload Excel File")
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.warning("Please upload an Excel file to proceed.")
    st.stop()

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
            "Escalations": {"Feeder": preserve_exact_string, "DT Nomenclature": preserve_exact_string, "Account No": preserve_exact_string}
        }
    )
except Exception as e:
    st.error(f"Error reading Excel file: {e}")
    st.stop()

# Load sheets
feeder_df = sheets.get("Feeder Data")
dt_df = sheets.get("Transformer Data")
ppm_df = sheets.get("Customer Data_PPM")
ppd_df = sheets.get("Customer Data_PPD")
band_df = sheets.get("Feeder Band")
tariff_df = sheets.get("Customer Tariffs")
escalations_df = sheets.get("Escalations")

if any(df is None for df in [feeder_df, dt_df, ppm_df, ppd_df, band_df, tariff_df, escalations_df]):
    st.error("One or more sheets missing.")
    st.stop()

# Validate columns (similar checks as original)
required_customer_cols = ["NAME_OF_DT", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "NAME_OF_FEEDER", "BUSINESS_UNIT", "UNDERTAKING", "TARIFF"]
required_dt_cols = ["New Unique DT Nomenclature", "Ownership", "Connection Status"]
required_feeder_cols = ["Feeder"]
required_band_cols = ["Feeder", "BAND"]
required_tariff_cols = ["Tariff"]
required_escalations_cols = ["Feeder", "DT Nomenclature", "Account No"]
checks = [
    (ppm_df, "Customer Data_PPM", required_customer_cols),
    (ppd_df, "Customer Data_PPD", required_customer_cols),
    (dt_df, "Transformer Data", required_dt_cols),
    (feeder_df, "Feeder Data", required_feeder_cols),
    (band_df, "Feeder Band", required_band_cols),
    (tariff_df, "Customer Tariffs", required_tariff_cols),
    (escalations_df, "Escalations", required_escalations_cols)
]
for df, name, cols in checks:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        if name == "Transformer Data" and "Ownership" in missing:
            st.warning("Ownership column missing in Transformer Data. Assuming PUBLIC")
            dt_df["Ownership"] = "PUBLIC"
        else:
            st.error(f"Missing columns in {name}: {missing}")
            st.stop()

# Months - adjust as needed (your original used JAN..JUN)
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
month_indices = {m:i for i,m in enumerate(months)}

# Fill missing months in relevant sheets
for df, name in [(ppm_df, "Customer Data_PPM"), (ppd_df, "Customer Data_PPD"), (feeder_df, "Feeder Data"), (dt_df, "Transformer Data")]:
    missing_months = [m for m in months if m not in df.columns]
    for m in missing_months:
        st.warning(f"Filling missing month {m} in {name} with zeros.")
        df[m] = 0

# Ensure tariff columns exist
for df, col in [(ppm_df, "TARIFF"), (ppd_df, "TARIFF"), (band_df, "BAND"), (tariff_df, "Tariff")]:
    if col not in df.columns:
        df[col] = ""

if "Short Name" not in band_df.columns:
    band_df["Short Name"] = band_df["Feeder"].apply(lambda x: get_short_name(x))

# Tariff rate handling (fallback to 209.5)
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

# merge feeders to bands
feeder_df = feeder_df.merge(band_df[["Feeder","BAND"]], on="Feeder", how="left")
feeder_df["BAND"] = feeder_df["BAND"].fillna("Unknown")
feeder_df["Tariff_Rate"] = feeder_df["BAND"].map(band_rates).fillna(209.5)
if feeder_df["BAND"].str.contains("Unknown").any():
    st.warning("Some feeders not mapped to bands; default tariff 209.5 used.")

# Convert month columns to kWh columns
for month in months:
    for df, unit in [(feeder_df, 1000), (ppm_df, 1), (ppd_df, 1)]:
        col = f"{month} (kWh)"
        df[col] = pd.to_numeric(df.get(month, 0), errors="coerce").fillna(0) * unit
    dt_df[f"{month} (kWh)"] = pd.to_numeric(dt_df.get(month, 0), errors="coerce").fillna(0)

# Drop original month columns
for df in [feeder_df, dt_df, ppm_df, ppd_df]:
    for m in months:
        if m in df.columns:
            df.drop(columns=[m], inplace=True)

# Filter NOT CONNECTED DTs with zero energy
dt_df["total_energy_kwh"] = dt_df[[f"{m} (kWh)" for m in months]].sum(axis=1)
not_connected_zero = (dt_df["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_df["total_energy_kwh"] == 0)
excluded_dts = dt_df[not_connected_zero]
if not excluded_dts.empty:
    st.warning(f"Excluding {len(excluded_dts)} DTs marked NOT CONNECTED with zero energy.")
dt_df = dt_df[~not_connected_zero].copy()

# Normalize names across sheets
norm_pairs = [
    ("Feeder", feeder_df), ("NAME_OF_FEEDER", ppm_df), ("NAME_OF_FEEDER", ppd_df), ("Feeder", band_df),
    ("NAME_OF_DT", ppm_df), ("NAME_OF_DT", ppd_df), ("New Unique DT Nomenclature", dt_df),
    ("TARIFF", ppm_df), ("TARIFF", ppd_df), ("Tariff", tariff_df),
    ("BUSINESS_UNIT", ppm_df), ("BUSINESS_UNIT", ppd_df),
    ("UNDERTAKING", ppm_df), ("UNDERTAKING", ppd_df),
    ("Feeder", escalations_df), ("DT Nomenclature", escalations_df)
]
for col, df in norm_pairs:
    if col in df.columns:
        df[col] = df[col].apply(normalize_name)

# Combine PPM and PPD
ppm_df["Billing_Type"] = "PPM"
ppd_df["Billing_Type"] = "PPD"
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True)
if customer_df.empty:
    st.error("No customers after combining PPM and PPD.")
    st.stop()

# Filter customers to valid DTs based on Transformer Data
valid_dts = set(dt_df["New Unique DT Nomenclature"])
customer_invalid_dts = customer_df[~customer_df["NAME_OF_DT"].isin(valid_dts)]
error_report = []
if not customer_invalid_dts.empty:
    for _, r in customer_invalid_dts.iterrows():
        error_report.append({
            "ACCOUNT_NUMBER": r.get("ACCOUNT_NUMBER"),
            "NAME_OF_DT": r.get("NAME_OF_DT"),
            "Reason": "NAME_OF_DT not in Transformer Data"
        })
error_report_df = pd.DataFrame(error_report)
customer_df = customer_df[customer_df["NAME_OF_DT"].isin(valid_dts)].copy()
if customer_df.empty:
    st.error("No valid customers after filtering for DTs.")
    st.stop()

# Create short names and Feeder link
feeder_df["Feeder_Short"] = feeder_df["Feeder"].apply(get_short_name)
dt_df["DT_Short_Name"] = dt_df["New Unique DT Nomenclature"].apply(lambda x: get_short_name(x, is_dt=True))
dt_df["Feeder"] = dt_df["New Unique DT Nomenclature"].apply(lambda x: "-".join(x.split("-")[:-1]) if isinstance(x,str) and "-" in x and len(x.split("-"))>=3 else x)
dt_df["Feeder"] = dt_df["Feeder"].apply(normalize_name)
dt_df["NAME_OF_DT"] = dt_df["New Unique DT Nomenclature"]
customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))
customer_df["Feeder"] = customer_df["NAME_OF_DT"].apply(lambda x: "-".join(x.split("-")[:-1]) if isinstance(x,str) and "-" in x and len(x.split("-"))>=3 else x)
customer_df["Feeder"] = customer_df["Feeder"].apply(normalize_name)

# Filter for valid feeders
valid_feeders = set(feeder_df["Feeder"])
dt_df = dt_df[dt_df["Feeder"].isin(valid_feeders)].copy()
customer_df = customer_df[customer_df["Feeder"].isin(valid_feeders)].copy()
if dt_df.empty or customer_df.empty:
    st.error("No valid data after filtering for Feeders.")
    st.stop()

# Map DT to feeder tariff rate
dt_df = dt_df.merge(feeder_df[["Feeder","Tariff_Rate"]], on="Feeder", how="left")
dt_df["Tariff_Rate"] = dt_df["Tariff_Rate"].fillna(209.5)

# Merge tariff rates into customers
tariff_matches = customer_df["TARIFF"].isin(tariff_df["Tariff"])
if not tariff_matches.all():
    st.warning(f"Some TARIFF values not found in Customer Tariffs.")
customer_df = customer_df.merge(tariff_df[["Tariff","Rate (NGN)"]], left_on="TARIFF", right_on="Tariff", how="left")
customer_df["Rate (NGN)"] = customer_df["Rate (NGN)"].fillna(209.5)
customer_df = customer_df.drop(columns=["Tariff"], errors="ignore")

# Escalations location trust scores
escalations_df["Report_Count"] = 1
feeder_escalations = escalations_df.groupby("Feeder")["Report_Count"].sum().reset_index()
feeder_escalations["location_trust_score"] = feeder_escalations["Report_Count"] / feeder_escalations["Report_Count"].max()
feeder_escalations["location_trust_score"] = feeder_escalations["location_trust_score"].fillna(0).clip(0,1)

dt_escalations = escalations_df.groupby("DT Nomenclature")["Report_Count"].sum().reset_index()
dt_escalations["location_trust_score"] = dt_escalations["Report_Count"] / dt_escalations["Report_Count"].max()
dt_escalations["location_trust_score"] = dt_escalations["location_trust_score"].fillna(0).clip(0,1)

# ------------------------
# UI Filters (kept same)
# ------------------------
st.title("Ikeja Electric Energy Theft Detection Dashboard")
st.subheader("Filters")
col1,col2,col3,col4,col5,col6 = st.columns(6)
with col1:
    bu_options = sorted(customer_df["BUSINESS_UNIT"].unique())
    selected_bu = st.selectbox("Select Business Unit", bu_options) if bu_options else ""
with col2:
    if selected_bu:
        customer_df_bu = customer_df[customer_df["BUSINESS_UNIT"] == selected_bu]
        ut_options = sorted(customer_df_bu["UNDERTAKING"].unique())
        selected_ut = st.selectbox("Select Undertaking", ut_options) if ut_options else ""
    else:
        selected_ut = ""
        customer_df_bu = pd.DataFrame()
with col3:
    if selected_ut:
        customer_df_ut = customer_df_bu[customer_df_bu["UNDERTAKING"] == selected_ut]
        feeder_options = sorted(feeder_df["Feeder_Short"].unique())
        selected_feeder_short = st.selectbox("Select Feeder", feeder_options) if feeder_options else ""
    else:
        selected_feeder_short = ""
        customer_df_ut = pd.DataFrame()
with col4:
    if selected_feeder_short:
        selected_feeder = feeder_df[feeder_df["Feeder_Short"]==selected_feeder_short]["Feeder"].iloc[0]
        dt_df_filtered = dt_df[dt_df["Feeder"]==selected_feeder]
        dt_options = sorted(dt_df_filtered["DT_Short_Name"].unique())
        selected_dt_short = st.selectbox("Select DT", dt_options) if dt_options else ""
    else:
        selected_feeder = ""
        selected_dt_short = ""
with col5:
    start_month = st.selectbox("Start Month", months)
with col6:
    end_month = st.selectbox("End Month", months, index=len(months)-1)
    if month_indices[start_month] > month_indices[end_month]:
        st.error("Start month must be before or equal to end month.")
        st.stop()

# ------------------------
# Weight sliders initialization (persist via session_state so optimizer can set them)
# ------------------------
# initialize session_state keys if not present
for key, val in [("w_feeder",0.2), ("w_dt",0.2), ("w_location",0.2), ("w_pattern",0.2), ("w_relative",0.2), ("w_zero", 0.05)]:
    if key not in st.session_state:
        st.session_state[key] = val

colw1, colw2, colw3 = st.columns(3)
with colw1:
    st.session_state["w_feeder"] = st.slider("Feeder Billing Efficiency Weight", 0.0, 1.0, st.session_state["w_feeder"], 0.01, key="w_feeder_slider")
    st.session_state["w_dt"] = st.slider("DT Billing Efficiency Weight", 0.0, 1.0, st.session_state["w_dt"], 0.01, key="w_dt_slider")
with colw2:
    st.session_state["w_location"] = st.slider("Location Trust Score Weight", 0.0, 1.0, st.session_state["w_location"], 0.01, key="w_location_slider")
    st.session_state["w_pattern"] = st.slider("Consumption Pattern Deviation Weight", 0.0, 1.0, st.session_state["w_pattern"], 0.01, key="w_pattern_slider")
with colw3:
    st.session_state["w_relative"] = st.slider("DT Relative Usage Score Weight", 0.0, 1.0, st.session_state["w_relative"], 0.01, key="w_relative_slider")
    st.session_state["w_zero"] = st.slider("Zero Frequency Weight", 0.0, 1.0, st.session_state["w_zero"], 0.01, key="w_zero_slider")

# ------------------------
# Prepare the monthly melted customer_monthly used for scoring
# ------------------------
# required id_vars
required_id_vars = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type", "Feeder", "Rate (NGN)"]
value_vars = [f"{m} (kWh)" for m in months]
missing_id_vars = [c for c in required_id_vars if c not in customer_df.columns]
if missing_id_vars:
    st.error(f"Missing id vars in customer data: {missing_id_vars}")
    st.stop()

try:
    customer_monthly = customer_df.melt(id_vars=required_id_vars, value_vars=value_vars, var_name="month", value_name="billed_kwh")
    customer_monthly["month"] = customer_monthly["month"].str.replace(" (kWh)", "")
    customer_monthly["month"] = pd.Categorical(customer_monthly["month"], categories=months, ordered=True)
except Exception as e:
    st.error(f"Failed to melt customer data: {e}")
    st.stop()

# ------------------------
# DT monthly melt (for dt_billing_efficiency)
# ------------------------
try:
    dt_agg_monthly = dt_df.melt(
        id_vars=["New Unique DT Nomenclature", "DT_Short_Name", "Feeder", "Tariff_Rate", "Ownership", "Connection Status", "total_energy_kwh"],
        value_vars=[f"{m} (kWh)" for m in months],
        var_name="month", value_name="total_dt_kwh"
    )
    dt_agg_monthly["month"] = dt_agg_monthly["month"].str.replace(" (kWh)", "")
    dt_agg_monthly["month"] = pd.Categorical(dt_agg_monthly["month"], categories=months, ordered=True)
except Exception as e:
    st.error(f"DT melt failed: {e}")
    st.stop()

# ------------------------
# Filter by selected months
# ------------------------
selected_months = months[month_indices[start_month] : month_indices[end_month] + 1]
customer_monthly = customer_monthly[customer_monthly["month"].isin(selected_months)].copy()
dt_agg_monthly = dt_agg_monthly[dt_agg_monthly["month"].isin(selected_months)].copy()
if customer_monthly.empty or dt_agg_monthly.empty:
    st.error("No data for selected months.")
    st.stop()

# ------------------------
# Compute dt_relative_usage
# ------------------------
dt_relative_usage = calculate_dt_relative_usage(customer_monthly, selected_months)

# ------------------------
# Aggregate for DT and Feeder (per your original process)
# ------------------------
period_label = f"{start_month}" if start_month == end_month else f"{start_month} to {end_month}"
try:
    customer_agg = customer_monthly.groupby(["NAME_OF_DT","Feeder"])["billed_kwh"].sum().reset_index().rename(columns={"billed_kwh":"customer_billed_kwh"})
    dt_agg_sum = dt_agg_monthly.groupby(["New Unique DT Nomenclature","DT_Short_Name","Feeder","Tariff_Rate","Ownership","Connection Status","total_energy_kwh"])["total_dt_kwh"].sum().reset_index()
except Exception as e:
    st.error(f"Aggregation failed: {e}")
    st.stop()

# DT billing efficiency (per original logic)
try:
    dt_merged = dt_agg_sum.merge(customer_agg, left_on=["New Unique DT Nomenclature","Feeder"], right_on=["NAME_OF_DT","Feeder"], how="left")
    dt_merged["customer_billed_kwh"] = dt_merged["customer_billed_kwh"].fillna(0)
    dt_merged["total_billed_kwh"] = np.where(dt_merged["Ownership"].str.strip().str.upper().isin(["PRIVATE"]), dt_merged["total_dt_kwh"], dt_merged["customer_billed_kwh"])
    dt_merged["dt_billing_efficiency"] = np.where(
        (dt_merged["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_merged["total_energy_kwh"] > 0),
        0.0,
        (dt_merged["total_billed_kwh"] / dt_merged["total_dt_kwh"].replace(0,1)).clip(0,1)
    )
    dt_merged["energy_lost_kwh"] = dt_merged["total_dt_kwh"] - dt_merged["total_billed_kwh"]
    dt_merged["financial_loss_naira"] = dt_merged["energy_lost_kwh"] * dt_merged["Tariff_Rate"]
except Exception as e:
    st.error(f"DT merge failed: {e}")
    st.stop()

# Per-month DT billing efficiency
try:
    customer_billed_monthly = customer_monthly.groupby(["NAME_OF_DT","month"])["billed_kwh"].sum().reset_index().rename(columns={"billed_kwh":"customer_billed_kwh"})
    dt_merged_monthly = dt_agg_monthly.merge(customer_billed_monthly, left_on=["New Unique DT Nomenclature","month"], right_on=["NAME_OF_DT","month"], how="left")
    dt_merged_monthly["customer_billed_kwh"] = dt_merged_monthly["customer_billed_kwh"].fillna(0)
    dt_merged_monthly["total_billed_kwh"] = np.where(dt_merged_monthly["Ownership"].str.strip().str.upper().isin(["PRIVATE"]), dt_merged_monthly["total_dt_kwh"], dt_merged_monthly["customer_billed_kwh"])
    dt_merged_monthly["dt_billing_efficiency"] = np.where(
        (dt_merged_monthly["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_merged_monthly["total_energy_kwh"] > 0),
        0.0,
        (dt_merged_monthly["total_billed_kwh"] / dt_merged_monthly["total_dt_kwh"].replace(0,1)).clip(0,1)
    )
except Exception as e:
    st.error(f"DT monthly merge failed: {e}")
    st.stop()

# Feeder consumption and feeder_merged
try:
    feeder_monthly = feeder_df.melt(id_vars=["Feeder","Feeder_Short","Tariff_Rate"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="feeder_energy_kwh")
    feeder_monthly["month"] = feeder_monthly["month"].str.replace(" (kWh)","")
    feeder_monthly["month"] = pd.Categorical(feeder_monthly["month"], categories=months, ordered=True)
    feeder_monthly = feeder_monthly[feeder_monthly["month"].isin(selected_months)]
    feeder_agg = feeder_monthly.groupby(["Feeder","Feeder_Short","Tariff_Rate"])["feeder_energy_kwh"].sum().reset_index()

    feeder_agg_billed = dt_merged.groupby(["Feeder"])["total_billed_kwh"].sum().reset_index()
    feeder_merged = feeder_agg.merge(feeder_agg_billed, on=["Feeder"], how="left")
    feeder_merged["total_billed_kwh"] = feeder_merged["total_billed_kwh"].fillna(0)
    feeder_merged["feeder_billing_efficiency"] = (feeder_merged["total_billed_kwh"] / feeder_merged["feeder_energy_kwh"].replace(0,1)).clip(0,1)
    feeder_merged["feeder_energy_lost_kwh"] = feeder_merged["feeder_energy_kwh"] - feeder_merged["total_billed_kwh"]
    feeder_merged["feeder_financial_loss_naira"] = feeder_merged["feeder_energy_lost_kwh"] * feeder_merged["Tariff_Rate"]
except Exception as e:
    st.error(f"Feeder processing failed: {e}")
    st.stop()

# Merge location trust scores into feeder and dt
feeder_merged = feeder_merged.merge(feeder_escalations[["Feeder","location_trust_score"]], on="Feeder", how="left")
feeder_merged["location_trust_score"] = feeder_merged["location_trust_score"].fillna(0)

dt_merged = dt_merged.merge(dt_escalations[["DT Nomenclature","location_trust_score"]], left_on="New Unique DT Nomenclature", right_on="DT Nomenclature", how="left")
dt_merged["location_trust_score"] = dt_merged["location_trust_score"].fillna(0)

dt_merged_monthly = dt_merged_monthly.merge(dt_escalations[["DT Nomenclature","location_trust_score"]], left_on="New Unique DT Nomenclature", right_on="DT Nomenclature", how="left")
dt_merged_monthly["location_trust_score"] = dt_merged_monthly["location_trust_score"].fillna(0)

# ------------------------
# Merge customer_monthly with feeder & dt monthly metrics
# ------------------------
customer_monthly = customer_monthly.merge(feeder_merged[["Feeder","feeder_billing_efficiency","location_trust_score"]], on="Feeder", how="left")
customer_monthly = customer_monthly.merge(
    dt_merged_monthly[["New Unique DT Nomenclature","month","dt_billing_efficiency","location_trust_score"]],
    left_on=["NAME_OF_DT","month"],
    right_on=["New Unique DT Nomenclature","month"],
    how="left"
)
# location trust: prefer customer-level feeder location then dt location
customer_monthly["location_trust_score"] = customer_monthly["location_trust_score_x"].combine_first(customer_monthly["location_trust_score_y"]).fillna(0)
customer_monthly["feeder_billing_efficiency"] = customer_monthly["feeder_billing_efficiency"].fillna(0)
customer_monthly["dt_billing_efficiency"] = customer_monthly["dt_billing_efficiency"].fillna(0)

# ------------------------
# Pattern and Zero scores (computed across full-customer rows)
# ------------------------
pattern_df = calculate_pattern_deviation(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
zero_df = calculate_zero_counter(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
# merge into customer_monthly (left join by account)
customer_monthly = customer_monthly.merge(pattern_df, left_on="ACCOUNT_NUMBER", right_on="id", how="left")
customer_monthly = customer_monthly.merge(zero_df, left_on="ACCOUNT_NUMBER", right_on="id", how="left")
customer_monthly["pattern_deviation_score"] = customer_monthly["pattern_deviation_score"].fillna(0)
customer_monthly["zero_counter_score"] = customer_monthly["zero_counter_score"].fillna(0)

# Merge dt_relative_usage
customer_monthly = customer_monthly.merge(dt_relative_usage, on="ACCOUNT_NUMBER", how="left")
customer_monthly["dt_relative_usage_score"] = customer_monthly["dt_relative_usage_score"].fillna(0)

# ------------------------
# Optimization section: if user clicks optimize -> adjust sliders in session_state and recompute
# ------------------------
if st.button("Optimize Customer-Level Weights for Escalations (grid step=0.05)"):
    escal_accounts = set(escalations_df["Account No"].astype(str).str.strip().tolist())
    # Ensure the subset used by optimizer has the merged fields
    optimizer_input = customer_monthly.groupby("ACCOUNT_NUMBER").agg({
        "feeder_billing_efficiency": "mean",
        "dt_billing_efficiency": "mean",
        "location_trust_score": "mean",
        "pattern_deviation_score": "mean",
        "dt_relative_usage_score": "mean",
        "zero_counter_score": "mean"
    }).reset_index()
    # rename columns to match expectation of optimizer (they already match)
    wp_n, wr_n, wz_n = optimize_customer_weights(optimizer_input, escal_accounts, st.session_state["w_feeder"], st.session_state["w_dt"], st.session_state["w_location"])
    # write optimized customer-level sub-weights back into session_state sliders
    st.session_state["w_pattern"] = float(np.round(wp_n, 2))
    st.session_state["w_relative"] = float(np.round(wr_n, 2))
    st.session_state["w_zero"] = float(np.round(wz_n, 2))
    # also update slider widgets by rerunning the app (so displayed slider positions update)
    st.experimental_rerun()

# ------------------------
# Normalize all weights to sum = 1 (so theft probability is weighted combination)
# ------------------------
w_feeder = st.session_state["w_feeder"]
w_dt = st.session_state["w_dt"]
w_location = st.session_state["w_location"]
w_pattern = st.session_state["w_pattern"]
w_relative = st.session_state["w_relative"]
w_zero = st.session_state["w_zero"]

total_w = w_feeder + w_dt + w_location + w_pattern + w_relative + w_zero
if total_w == 0:
    st.error("Total weight cannot be zero.")
    st.stop()

w_feeder /= total_w
w_dt /= total_w
w_location /= total_w
w_pattern /= total_w
w_relative /= total_w
w_zero /= total_w

# ------------------------
# Theft probability calculation at customer-month level
# ------------------------
customer_monthly["theft_probability"] = (
    w_feeder * (1 - customer_monthly["feeder_billing_efficiency"]) +
    w_dt * (1 - customer_monthly["dt_billing_efficiency"]) +
    w_location * customer_monthly["location_trust_score"] +
    w_pattern * customer_monthly["pattern_deviation_score"] +
    w_relative * customer_monthly["dt_relative_usage_score"] +
    w_zero * customer_monthly["zero_counter_score"]
).clip(0,1)

# risk tiers
customer_monthly["risk_tier"] = pd.cut(customer_monthly["theft_probability"], bins=[-1, 0.4, 0.7, 1.0], labels=["Low","Medium","High"])

# ------------------------
# CUSTOMER / DT / FEEDER UI outputs (kept similar to your original)
# ------------------------
# quick customer list button (by selected DT)
st.subheader(f"Customers under {selected_dt_short} ({selected_feeder_short}, {period_label})")
if selected_dt_short and st.button("Show Customer List"):
    filtered_customers = customer_monthly[customer_monthly["DT_Short_Name"] == selected_dt_short].copy()
    if filtered_customers.empty:
        st.warning("No customers for this DT.")
    else:
        month_customers = filtered_customers.groupby(["ACCOUNT_NUMBER","METER_NUMBER","CUSTOMER_NAME","ADDRESS","Billing_Type"]).agg({
            "billed_kwh":"sum",
            "theft_probability":"mean",
            "risk_tier": lambda x: pd.Series(x).mode()[0] if not x.mode().empty else "Unknown",
            "pattern_deviation_score":"mean",
            "dt_relative_usage_score":"mean",
            "zero_counter_score":"mean"
        }).reset_index()
        display_columns = ["ACCOUNT_NUMBER","METER_NUMBER","CUSTOMER_NAME","ADDRESS","billed_kwh","Billing_Type","theft_probability","risk_tier","pattern_deviation_score","dt_relative_usage_score","zero_counter_score"]
        month_customers = month_customers.sort_values(by="theft_probability", ascending=False)
        st.dataframe(month_customers[display_columns].style.format({
            "billed_kwh":"{:.2f}",
            "theft_probability":"{:.3f}",
            "pattern_deviation_score":"{:.3f}",
            "dt_relative_usage_score":"{:.3f}",
            "zero_counter_score":"{:.3f}"
        }), use_container_width=True)

# DT heatmap (same idea)
st.subheader("DT Theft Probability Heatmap")
if selected_feeder:
    filtered_dt_agg = dt_merged_monthly[dt_merged_monthly["Feeder"] == selected_feeder].copy()
    if filtered_dt_agg.empty:
        st.warning("No DT data for selected feeder.")
    else:
        dt_theft_scores = filtered_dt_agg.groupby("DT_Short_Name")["dt_billing_efficiency"].mean().reset_index()
        dt_theft_scores["theft_probability"] = 1 - dt_theft_scores["dt_billing_efficiency"]
        dt_order = dt_theft_scores.sort_values("theft_probability", ascending=False)["DT_Short_Name"].tolist()
        dt_pivot = filtered_dt_agg.pivot_table(index="DT_Short_Name", columns="month", values="dt_billing_efficiency", aggfunc="mean").reindex(index=dt_order, columns=months)
        if not dt_pivot.empty:
            plt.figure(figsize=(10,8))
            sns.heatmap(1 - dt_pivot, cmap="YlOrRd", vmin=0, vmax=1, cbar_kws={"label":"DT Theft Probability"})
            plt.title(f"DT Theft Probability for {selected_feeder_short} ({period_label})")
            st.pyplot(plt.gcf())
            plt.close()

# Feeder summary button
st.subheader("Feeder Summary")
if st.button("Show Feeder Summary"):
    try:
        cols = ["Feeder_Short","feeder_energy_kwh","total_billed_kwh","feeder_energy_lost_kwh","feeder_financial_loss_naira","feeder_billing_efficiency","location_trust_score"]
        missing = [c for c in cols if c not in feeder_merged.columns]
        if missing:
            st.error(f"Missing feeder columns: {missing}")
        else:
            fsum = feeder_merged[cols].copy()
            fsum["Period"] = period_label
            fsum.columns = ["Feeder","Energy Supplied (kWh)","Energy Billed (kWh)","Energy Unaccounted For (kWh)","Financial Loss (NGN)","Billing Efficiency","Location Trust Score","Period"]
            st.dataframe(fsum.style.format({
                "Energy Supplied (kWh)": "{:.2f}",
                "Energy Billed (kWh)": "{:.2f}",
                "Energy Unaccounted For (kWh)": "{:.2f}",
                "Financial Loss (NGN)": "{:.2f}",
                "Billing Efficiency": "{:.3f}",
                "Location Trust Score": "{:.3f}"
            }), use_container_width=True)
    except Exception as e:
        st.error(f"Feeder summary failed: {e}")

# DT summary button
st.subheader(f"DT Summary for {selected_feeder_short}")
if selected_feeder_short and st.button("Show DT Summary"):
    try:
        dt_summary = dt_merged[dt_merged["Feeder"] == selected_feeder].groupby(["DT_Short_Name"]).agg({
            "total_dt_kwh":"sum",
            "total_billed_kwh":"sum",
            "energy_lost_kwh":"sum",
            "financial_loss_naira":"sum",
            "dt_billing_efficiency":"mean",
            "location_trust_score":"mean"
        }).reset_index()
        dt_summary["Period"] = period_label
        dt_summary.columns = ["DT","Energy Supplied (kWh)","Energy Billed (kWh)","Energy Unaccounted For (kWh)","Financial Loss (NGN)","Billing Efficiency","Location Trust Score","Period"]
        st.dataframe(dt_summary.style.format({
            "Energy Supplied (kWh)":"{:.2f}",
            "Energy Billed (kWh)":"{:.2f}",
            "Energy Unaccounted For (kWh)":"{:.2f}",
            "Financial Loss (NGN)":"{:.2f}",
            "Billing Efficiency":"{:.3f}",
            "Location Trust Score":"{:.3f}"
        }), use_container_width=True)
    except Exception as e:
        st.error(f"DT summary failed: {e}")

# ------------------------
# Escalations report generation (independent of filters)
# ------------------------
st.subheader("Escalations Report")
st.markdown("This report checks every 'Account No' listed in the Escalations sheet against **both** prepaid and postpaid customer lists and returns the customer's monthly readings and the **full weighted theft probability** assigned (across selected months). Accounts not found are marked 'Found=False'.")

if st.button("Generate Escalations Report"):
    try:
        # Use the raw ppm_df and ppd_df (not filtered) to find account matches
        report_df = generate_escalations_report(ppm_df, ppd_df, escalations_df, customer_monthly)
        st.success(f"Escalations report generated ({len(report_df)} rows).")
        st.dataframe(report_df, use_container_width=True)

        # prepare excel for download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            report_df.to_excel(writer, index=False, sheet_name="Escalations Report")
        buffer.seek(0)

        st.download_button(
            label="📥 Download Escalations Report (Excel)",
            data=buffer,
            file_name=f"Escalations_Report_{period_label.replace(' ','_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Failed to generate escalations report: {e}")

# Footer
st.markdown("Built by Elvis Ebenuwah for Ikeja Electric. 2025.")
