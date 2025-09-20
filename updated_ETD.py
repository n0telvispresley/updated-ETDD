# updated_ETD_full.py
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Streamlit page
st.set_page_config(page_title="IE Energy Theft Detection Dashboard", layout="wide")

# -----------------------
# Utility / Helper funcs
# -----------------------
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

# Pattern deviation: improved (flags months < 60% of max)
def calculate_pattern_deviation(df, id_col, value_cols):
    rows = []
    valid_cols = [c for c in value_cols if c in df.columns]
    for id_val, group in df.groupby(id_col):
        # take first row for this id (data shaped like one row per customer)
        values = group[valid_cols].iloc[0].values.astype(float)
        max_val = values.max() if len(values) > 0 else 0.0
        if max_val == 0:
            # if always zero, suspicious => highest score
            score = 1.0
        else:
            below = np.sum(values < 0.6 * max_val)
            score = below / len(valid_cols)
        rows.append({"id": id_val, "pattern_deviation_score": min(score, 1.0)})
    return pd.DataFrame(rows)

# Zero counter: fraction of months with zero reading
def calculate_zero_counter(df, id_col, value_cols):
    rows = []
    valid_cols = [c for c in value_cols if c in df.columns]
    for id_val, group in df.groupby(id_col):
        values = group[valid_cols].iloc[0].values.astype(float)
        zeros = np.sum(values == 0)
        score = zeros / len(valid_cols) if len(valid_cols) > 0 else 0.0
        rows.append({"id": id_val, "zero_counter_score": min(score, 1.0)})
    return pd.DataFrame(rows)

# DT relative usage score: modified to ignore zeros when averaging DT
def calculate_dt_relative_usage(customer_monthly, selected_months):
    # customer_monthly expected columns: ACCOUNT_NUMBER, NAME_OF_DT, month, billed_kwh
    # Use only selected_months
    cm = customer_monthly[customer_monthly["month"].isin(selected_months)].copy()
    # Aggregate by account & DT
    customer_agg = cm.groupby(["ACCOUNT_NUMBER", "NAME_OF_DT"])["billed_kwh"].sum().reset_index()
    # compute DT average ignoring zero customers (i.e., for dt avg compute average of customers with billed_kwh>0)
    # but since we aggregated summed over months, we want per-account total >0 included
    dt_avg = customer_agg[customer_agg["billed_kwh"] > 0].groupby("NAME_OF_DT")["billed_kwh"].mean().reset_index().rename(columns={"billed_kwh": "dt_avg_kwh"})
    customer_agg = customer_agg.merge(dt_avg, on="NAME_OF_DT", how="left")
    customer_agg["dt_avg_kwh"] = customer_agg["dt_avg_kwh"].fillna(0)
    # Avoid division by zero
    customer_agg["relative_ratio"] = np.where(customer_agg["dt_avg_kwh"] == 0, 0.5, customer_agg["billed_kwh"] / customer_agg["dt_avg_kwh"])
    # Now compute dt_relative_usage_score similar to your original mapping
    def rel_score(row):
        try:
            if row["dt_avg_kwh"] == 0:
                # If DT average zero, we can't infer much — give neutral small score
                return 0.1
            if row["billed_kwh"] < row["dt_avg_kwh"] * 0.3:
                return 0.9
            if row["billed_kwh"] > row["dt_avg_kwh"] * 0.7:
                return 0.1
            # interpolate between 0.1 and 0.9
            rr = row["relative_ratio"]
            # rr in [0.3,0.7] maps to [0.9,0.1] decreasing
            return 0.1 + (0.9 - 0.1) * (0.7 - rr) / (0.7 - 0.3)
        except Exception:
            return 0.1
    customer_agg["dt_relative_usage_score"] = customer_agg.apply(rel_score, axis=1).clip(0,1)
    return customer_agg[["ACCOUNT_NUMBER", "dt_relative_usage_score"]]

# -----------------------
# Load Excel
# -----------------------
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

# Extract sheets
feeder_df = sheets.get("Feeder Data")
dt_df = sheets.get("Transformer Data")
ppm_df = sheets.get("Customer Data_PPM")
ppd_df = sheets.get("Customer Data_PPD")
band_df = sheets.get("Feeder Band")
tariff_df = sheets.get("Customer Tariffs")
escalations_df = sheets.get("Escalations")

# Validate existence
if any(df is None for df in [feeder_df, dt_df, ppm_df, ppd_df, band_df, tariff_df, escalations_df]):
    st.error("One or more required sheets missing.")
    st.stop()

# Required columns checks (same as original)
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
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        if name == "Transformer Data" and "Ownership" in missing_cols:
            st.warning("Ownership missing in Transformer Data. Assuming PUBLIC for all.")
            dt_df["Ownership"] = "PUBLIC"
        else:
            st.error(f"Missing columns in {name}: {missing_cols}")
            st.stop()

# Months (keep same)
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
month_indices = {m: i for i, m in enumerate(months)}

# Fill missing month cols with 0
for df, name in [(ppm_df, "Customer Data_PPM"), (ppd_df, "Customer Data_PPD"), (feeder_df, "Feeder Data"), (dt_df, "Transformer Data")]:
    missing = [m for m in months if m not in df.columns]
    if missing:
        st.warning(f"Filling missing months in {name}: {missing}")
        for m in missing:
            df[m] = 0

# Ensure some optional columns exist
for df, col in [(ppm_df, "TARIFF"), (ppd_df, "TARIFF"), (band_df, "BAND"), (tariff_df, "Tariff")]:
    if col not in df.columns:
        df[col] = ""

if "Short Name" not in band_df.columns:
    band_df["Short Name"] = band_df["Feeder"].apply(lambda x: get_short_name(x))

# Handle Rate column in tariff_df
rate_col = next((c for c in ["Rate (NGN)", "Rate (₦)", "Rate", "RATE", "Rate(NGN)", "Rate(₦)"] if c in tariff_df.columns), None)
if rate_col:
    tariff_df["Rate (NGN)"] = pd.to_numeric(tariff_df[rate_col], errors="coerce").fillna(209.5)
else:
    tariff_df["Rate (NGN)"] = 209.5

# Compute band rates
band_tariffs = {
    "A": ["A-MD1", "A-MD2", "A-Non MD"],
    "B": ["B-MD1", "B-MD2", "B-Non MD"],
    "C": ["C-MD1", "C-MD2", "C-Non MD"],
    "D": ["D-MD1", "D-MD2", "D-Non MD"],
    "E": ["E-MD1", "E-MD2", "E-Non MD"]
}
band_rates = {}
for band, tfs in band_tariffs.items():
    rates = tariff_df[tariff_df["Tariff"].isin(tfs)]["Rate (NGN)"]
    band_rates[band] = rates.mean() if not rates.empty else 209.5

# Map feeders to bands and tariff rate
feeder_df = feeder_df.merge(band_df[["Feeder", "BAND"]], on="Feeder", how="left")
feeder_df["BAND"] = feeder_df["BAND"].fillna("Unknown")
feeder_df["Tariff_Rate"] = feeder_df["BAND"].map(band_rates).fillna(209.5)
if feeder_df["BAND"].str.contains("Unknown").any():
    st.warning("Some feeders not mapped to bands. Using default tariff rate 209.5.")

# Data preprocessing: convert month cols to "{M} (kWh)" as in original
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

# drop original month columns to avoid confusion
for df in [feeder_df, dt_df, ppm_df, ppd_df]:
    df.drop(columns=months, errors="ignore", inplace=True)

# Filter NOT CONNECTED DTs with zero energy as original
dt_df["total_energy_kwh"] = dt_df[[f"{m} (kWh)" for m in months]].sum(axis=1)
not_connected_zero = (dt_df["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_df["total_energy_kwh"] == 0)
excluded_dts = dt_df[not_connected_zero][["New Unique DT Nomenclature", "Connection Status", "total_energy_kwh"]]
if not excluded_dts.empty:
    st.warning(f"Excluding {len(excluded_dts)} DTs marked NOT CONNECTED with zero energy.")
dt_df = dt_df[~not_connected_zero]

# Normalize many fields
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

# Combine PPM and PPD -> customer_df_all (the full customer list, used for escalations lookups)
ppm_df["Billing_Type"] = "PPM"
ppd_df["Billing_Type"] = "PPD"
customer_df_all = pd.concat([ppm_df, ppd_df], ignore_index=True)

if customer_df_all.empty:
    st.error("Combined customer dataset is empty.")
    st.stop()

# Filter for valid DTs from transformer data
valid_dts = set(dt_df["New Unique DT Nomenclature"])
customer_invalid_dts = customer_df_all[~customer_df_all["NAME_OF_DT"].isin(valid_dts)]
# Build error report if any invalid DTs
error_report = []
if not customer_invalid_dts.empty:
    for _, r in customer_invalid_dts.iterrows():
        error_report.append({
            "ACCOUNT_NUMBER": r.get("ACCOUNT_NUMBER"),
            "NAME_OF_DT": r.get("NAME_OF_DT"),
            "NAME_OF_FEEDER": r.get("NAME_OF_FEEDER"),
            "BUSINESS_UNIT": r.get("BUSINESS_UNIT"),
            "UNDERTAKING": r.get("UNDERTAKING"),
            "Reason": "NAME_OF_DT not in Transformer Data"
        })
error_report_df = pd.DataFrame(error_report)
# Keep only customers with valid DTs for the dashboard main operations
customer_df = customer_df_all[customer_df_all["NAME_OF_DT"].isin(valid_dts)].copy()
if customer_df.empty:
    st.error("No valid customers after filtering for Transformer Data DTs.")
    st.stop()

# Create short names and feeder links
feeder_df["Feeder_Short"] = feeder_df["Feeder"].apply(lambda x: get_short_name(x))
dt_df["DT_Short_Name"] = dt_df["New Unique DT Nomenclature"].apply(lambda x: get_short_name(x, is_dt=True))
# Build DT feeder link
dt_df["Feeder"] = dt_df["New Unique DT Nomenclature"].apply(
    lambda x: "-".join(x.split("-")[:-1]) if isinstance(x,str) and "-" in x and len(x.split("-"))>=3 else x
)
dt_df["Feeder"] = dt_df["Feeder"].apply(normalize_name)
dt_df["NAME_OF_DT"] = dt_df["New Unique DT Nomenclature"].copy()

customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))
customer_df["Feeder"] = customer_df["NAME_OF_DT"].apply(lambda x: "-".join(x.split("-")[:-1]) if isinstance(x,str) and "-" in x and len(x.split("-"))>=3 else x)
customer_df["Feeder"] = customer_df["Feeder"].apply(normalize_name)

# Filter for valid feeders
valid_feeders = set(feeder_df["Feeder"])
dt_df = dt_df[dt_df["Feeder"].isin(valid_feeders)].copy()
customer_df = customer_df[customer_df["Feeder"].isin(valid_feeders)].copy()
customer_df_all = customer_df_all[customer_df_all["Feeder"].isin(valid_feeders)].copy()

if dt_df.empty or customer_df.empty:
    st.error("No valid data after filtering for feeders.")
    st.stop()

# Map DT to tariff rate
dt_df = dt_df.merge(feeder_df[["Feeder", "Tariff_Rate"]], on="Feeder", how="left")
dt_df["Tariff_Rate"] = dt_df["Tariff_Rate"].fillna(209.5)

# Merge tariffs into customer_df
tariff_matches = customer_df["TARIFF"].isin(tariff_df["Tariff"])
if not tariff_matches.all():
    st.warning(f"Some customer TARIFF values not found in Customer Tariffs: {customer_df.loc[~tariff_matches, 'TARIFF'].unique()}")
customer_df = customer_df.merge(tariff_df[["Tariff", "Rate (NGN)"]], left_on="TARIFF", right_on="Tariff", how="left")
customer_df["Rate (NGN)"] = customer_df["Rate (NGN)"].fillna(209.5)
customer_df.drop(columns=["Tariff"], errors="ignore", inplace=True)

# Also prepare customer_df_all tarfff rates (for escalations lookup use)
customer_df_all = customer_df_all.merge(tariff_df[["Tariff", "Rate (NGN)"]], left_on="TARIFF", right_on="Tariff", how="left")
customer_df_all["Rate (NGN)"] = customer_df_all["Rate (NGN)"].fillna(209.5)
customer_df_all.drop(columns=["Tariff"], errors="ignore", inplace=True)

# Build Escalations-derived trust scores (as original)
escalations_df["Report_Count"] = 1
feeder_escalations = escalations_df.groupby("Feeder")["Report_Count"].sum().reset_index()
feeder_escalations["location_trust_score"] = feeder_escalations["Report_Count"] / feeder_escalations["Report_Count"].max()
feeder_escalations["location_trust_score"] = feeder_escalations["location_trust_score"].fillna(0).clip(0,1)

dt_escalations = escalations_df.groupby("DT Nomenclature")["Report_Count"].sum().reset_index()
dt_escalations["location_trust_score"] = dt_escalations["Report_Count"] / dt_escalations["Report_Count"].max()
dt_escalations["location_trust_score"] = dt_escalations["location_trust_score"].fillna(0).clip(0,1)

# --- UI Filters (original) ---
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

# --- Weight sliders (including new zero slider) ---
# Ensure session_state keys exist so we can update them safely later
if "w_pattern" not in st.session_state:
    st.session_state.w_pattern = 0.2
if "w_relative" not in st.session_state:
    st.session_state.w_relative = 0.2
if "w_zero" not in st.session_state:
    st.session_state.w_zero = 0.05  # default as requested

st.subheader("Adjust Theft Probability Weights")
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

# Normalize weights (prevent zero sum)
total_weight = w_feeder + w_dt + w_location + w_pattern + w_relative + w_zero
if total_weight == 0:
    st.error("Total weight cannot be zero. Please adjust sliders.")
    st.stop()
w_feeder /= total_weight
w_dt /= total_weight
w_location /= total_weight
w_pattern /= total_weight
w_relative /= total_weight
w_zero /= total_weight

# -----------------------
# Melt / monthly transforms (for dashboard) - operate on filtered customer_df (by BU/UT) for main visuals
# -----------------------
# Use filtered customer_df_ut if available for UI; else full customer_df
if 'customer_df_ut' not in locals() or customer_df_ut.empty:
    st.warning("No filtered customer data available. Using full dataset.")
    customer_df_filtered = customer_df
else:
    customer_df_filtered = customer_df_ut

# Build customer_monthly (for dashboards) from customer_df_filtered
required_id_vars = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type", "Feeder", "Rate (NGN)"]
value_vars = [f"{m} (kWh)" for m in months]
missing_id_vars = [c for c in required_id_vars if c not in customer_df_filtered.columns]
if missing_id_vars:
    st.error(f"Missing id_vars in customer_df_filtered: {missing_id_vars}")
    st.stop()

try:
    customer_monthly = customer_df_filtered.melt(id_vars=required_id_vars, value_vars=value_vars, var_name="month", value_name="billed_kwh")
    customer_monthly["month"] = customer_monthly["month"].str.replace(" (kWh)", "")
    customer_monthly["month"] = pd.Categorical(customer_monthly["month"], categories=months, ordered=True)
except Exception as e:
    st.error(f"Melt failed for customer_monthly: {e}")
    st.stop()

# Also build customer_monthly_all for escalations/optimizer (independent of filters)
required_id_vars_all = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type", "Feeder", "Rate (NGN)"]
missing_all = [c for c in required_id_vars_all if c not in customer_df_all.columns]
if missing_all:
    st.error(f"Missing id_vars in customer_df_all for escalations: {missing_all}")
    st.stop()
try:
    customer_monthly_all = customer_df_all.melt(id_vars=required_id_vars_all, value_vars=value_vars, var_name="month", value_name="billed_kwh")
    customer_monthly_all["month"] = customer_monthly_all["month"].str.replace(" (kWh)", "")
    customer_monthly_all["month"] = pd.Categorical(customer_monthly_all["month"], categories=months, ordered=True)
except Exception as e:
    st.error(f"Melt failed for customer_monthly_all: {e}")
    st.stop()

# DT monthly (for heatmaps) - use dt_df
try:
    dt_agg_monthly = dt_df.melt(
        id_vars=["New Unique DT Nomenclature", "DT_Short_Name", "Feeder", "Tariff_Rate", "Ownership", "Connection Status", "total_energy_kwh"],
        value_vars=[f"{m} (kWh)" for m in months],
        var_name="month",
        value_name="total_dt_kwh"
    )
    dt_agg_monthly["month"] = dt_agg_monthly["month"].str.replace(" (kWh)", "")
    dt_agg_monthly["month"] = pd.Categorical(dt_agg_monthly["month"], categories=months, ordered=True)
except Exception as e:
    st.error(f"DT melt failed: {e}")
    st.stop()

# -----------------------
# Filter by selected months for dashboard visuals
# -----------------------
selected_months = months[month_indices[start_month]: month_indices[end_month] + 1]
if not selected_months:
    st.error("No months selected.")
    st.stop()

customer_monthly = customer_monthly[customer_monthly["month"].isin(selected_months)]
customer_monthly_all = customer_monthly_all[customer_monthly_all["month"].isin(selected_months)]
dt_agg_monthly = dt_agg_monthly[dt_agg_monthly["month"].isin(selected_months)]

if customer_monthly.empty or customer_monthly_all.empty or dt_agg_monthly.empty:
    st.error(f"No data for selected months {selected_months}.")
    st.stop()

# -----------------------
# Compute DT relative usage (for dashboard) using customer_monthly (filtered) and customer_monthly_all for escalations
# -----------------------
dt_relative_usage = calculate_dt_relative_usage(customer_monthly, selected_months)
dt_relative_usage_all = calculate_dt_relative_usage(customer_monthly_all, selected_months)

# -----------------------
# Aggregate customer / dt for billing efficiencies (unchanged logic but careful with NaNs).
# -----------------------
period_label = f"{start_month}" if start_month == end_month else f"{start_month} to {end_month}"

# Aggregations (customer_agg & dt_agg_sum) for dashboard (filtered dataset)
try:
    cust_agg = customer_monthly.groupby(["NAME_OF_DT", "Feeder"])["billed_kwh"].sum().reset_index().rename(columns={"billed_kwh": "customer_billed_kwh"})
except Exception as e:
    st.error(f"Customer aggregation failed: {e}")
    st.stop()

try:
    dt_agg_sum = dt_agg_monthly.groupby(["New Unique DT Nomenclature", "DT_Short_Name", "Feeder", "Tariff_Rate", "Ownership", "Connection Status", "total_energy_kwh"])["total_dt_kwh"].sum().reset_index()
except Exception as e:
    st.error(f"DT aggregation failed: {e}")
    st.stop()

# DT billing efficiency (filtered)
try:
    dt_merged = dt_agg_sum.merge(cust_agg, left_on=["New Unique DT Nomenclature", "Feeder"], right_on=["NAME_OF_DT", "Feeder"], how="left")
    dt_merged["customer_billed_kwh"] = dt_merged["customer_billed_kwh"].fillna(0)
    dt_merged["total_billed_kwh"] = np.where(
        dt_merged["Ownership"].str.strip().str.upper().isin(["PRIVATE"]), dt_merged["total_dt_kwh"], dt_merged["customer_billed_kwh"]
    )
    dt_merged["dt_billing_efficiency"] = np.where(
        (dt_merged["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_merged["total_energy_kwh"] > 0),
        0.0,
        (dt_merged["total_billed_kwh"] / dt_merged["total_dt_kwh"].replace(0, 1)).clip(0, 1)
    )
    dt_merged["energy_lost_kwh"] = dt_merged["total_dt_kwh"] - dt_merged["total_billed_kwh"]
    dt_merged["financial_loss_naira"] = dt_merged["energy_lost_kwh"] * dt_merged["Tariff_Rate"]
except Exception as e:
    st.error(f"DT merge failed: {e}")
    st.stop()

# Per-month DT billing efficiency for heatmap (filtered)
try:
    customer_billed_monthly = customer_monthly.groupby(["NAME_OF_DT", "month"])["billed_kwh"].sum().reset_index().rename(columns={"billed_kwh":"customer_billed_kwh"})
    dt_merged_monthly = dt_agg_monthly.merge(customer_billed_monthly, left_on=["New Unique DT Nomenclature","month"], right_on=["NAME_OF_DT","month"], how="left")
    dt_merged_monthly["customer_billed_kwh"] = dt_merged_monthly["customer_billed_kwh"].fillna(0)
    dt_merged_monthly["total_billed_kwh"] = np.where(
        dt_merged_monthly["Ownership"].str.strip().str.upper().isin(["PRIVATE"]), dt_merged_monthly["total_dt_kwh"], dt_merged_monthly["customer_billed_kwh"]
    )
    dt_merged_monthly["dt_billing_efficiency"] = np.where(
        (dt_merged_monthly["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_merged_monthly["total_energy_kwh"] > 0),
        0.0,
        (dt_merged_monthly["total_billed_kwh"] / dt_merged_monthly["total_dt_kwh"].replace(0,1)).clip(0,1)
    )
except Exception as e:
    st.error(f"DT monthly merge failed: {e}")
    st.stop()

# Feeder monthly & billing efficiency (filtered)
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
    st.error(f"Feeder aggregation failed: {e}")
    st.stop()

# Merge location trust scores into feeder and dt merges
feeder_merged = feeder_merged.merge(feeder_escalations[["Feeder","location_trust_score"]], on="Feeder", how="left")
feeder_merged["location_trust_score"] = feeder_merged["location_trust_score"].fillna(0)
dt_merged = dt_merged.merge(dt_escalations[["DT Nomenclature","location_trust_score"]], left_on="New Unique DT Nomenclature", right_on="DT Nomenclature", how="left")
dt_merged["location_trust_score"] = dt_merged["location_trust_score"].fillna(0)
dt_merged_monthly = dt_merged_monthly.merge(dt_escalations[["DT Nomenclature","location_trust_score"]], left_on="New Unique DT Nomenclature", right_on="DT Nomenclature", how="left")
dt_merged_monthly["location_trust_score"] = dt_merged_monthly["location_trust_score"].fillna(0)

# -----------------------
# Pattern/zero scores and customer theft probability (filtered for dashboard)
# -----------------------
pattern_df = calculate_pattern_deviation(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
zero_df = calculate_zero_counter(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])

# Need customer_monthly to have feeder_billing_efficiency and dt_billing_efficiency per month
# Merge feeder and dt monthly metrics into customer_monthly (filtered)
customer_monthly["energy_billed_score"] = (1 - customer_monthly["billed_kwh"] / customer_monthly["billed_kwh"].replace(0,1).max()).clip(0,1)
customer_monthly = customer_monthly.merge(feeder_merged[["Feeder","feeder_billing_efficiency","location_trust_score"]], on="Feeder", how="left")
customer_monthly = customer_monthly.merge(dt_merged_monthly[["New Unique DT Nomenclature","month","dt_billing_efficiency","location_trust_score"]], left_on=["NAME_OF_DT","month"], right_on=["New Unique DT Nomenclature","month"], how="left")
customer_monthly = customer_monthly.merge(pattern_df, left_on=["ACCOUNT_NUMBER"], right_on=["id"], how="left")
customer_monthly = customer_monthly.merge(zero_df, left_on=["ACCOUNT_NUMBER"], right_on=["id"], how="left")
customer_monthly = customer_monthly.merge(dt_relative_usage, on="ACCOUNT_NUMBER", how="left")

# fillna defaults
customer_monthly["feeder_billing_efficiency"] = customer_monthly["feeder_billing_efficiency"].fillna(0)
customer_monthly["dt_billing_efficiency"] = customer_monthly["dt_billing_efficiency"].fillna(0)
# combine location trust source - feeder first then dt
customer_monthly["location_trust_score"] = customer_monthly["location_trust_score_x"].combine_first(customer_monthly["location_trust_score_y"]).fillna(0)
customer_monthly["pattern_deviation_score"] = customer_monthly["pattern_deviation_score"].fillna(0)
customer_monthly["zero_counter_score"] = customer_monthly["zero_counter_score"].fillna(0)
customer_monthly["dt_relative_usage_score"] = customer_monthly["dt_relative_usage_score"].fillna(0)

# Compute theft_probability for customer_monthly (filtered)
customer_monthly["theft_probability"] = (
    w_feeder * (1 - customer_monthly["feeder_billing_efficiency"]) +
    w_dt * (1 - customer_monthly["dt_billing_efficiency"]) +
    w_location * customer_monthly["location_trust_score"] +
    w_pattern * customer_monthly["pattern_deviation_score"] +
    w_relative * customer_monthly["dt_relative_usage_score"] +
    w_zero * customer_monthly["zero_counter_score"]
).clip(0,1)

customer_monthly["risk_tier"] = pd.cut(customer_monthly["theft_probability"], bins=[0,0.4,0.7,1.0], labels=["Low","Medium","High"], include_lowest=True)

# -----------------------
# NEW: optimizer that searches pattern/relative/zero weights (0..1 step 0.05) to maximize mean theft for escalations accounts
# Optimizes only the sub-weights among pattern, relative, zero while keeping feeder/dt/location as chosen by the user.
# -----------------------
def optimize_customer_weights(customer_monthly_df_all, escalations_df_local, w_feeder_local, w_dt_local, w_location_local):
    """
    Returns tuple ((best_pattern_w, best_relative_w, best_zero_w), pre_mean, post_mean)
    pre_mean: average theft across escalations with current weights
    post_mean: average theft after applying best combination
    """
    try:
        # accounts in escalations
        accounts = escalations_df_local["Account No"].astype(str).unique().tolist()
        # subset from the *ALL* customer monthly (customer_monthly_all-like DataFrame)
        subset = customer_monthly_df_all[customer_monthly_df_all["ACCOUNT_NUMBER"].astype(str).isin(accounts)].copy()
        if subset.empty:
            return (0.0, 0.0, 0.0), None, None

        # compute current mean theft with existing weights (we assume customer_monthly_df_all already has theft_probability computed with current slider weights)
        pre_mean = subset["theft_probability"].mean()

        best_combo = (0.0, 0.0, 0.0)
        best_mean = -np.inf

        # search grid (0..1 step 0.05)
        steps = np.arange(0.0, 1.0001, 0.05)
        for p in steps:
            for r in steps:
                for z in steps:
                    total = p + r + z
                    if total == 0:
                        continue
                    p_n, r_n, z_n = p / total, r / total, z / total
                    # compute theft scores using provided feeder/dt/location weights (unchanged)
                    theft_scores = (
                        w_feeder_local * (1 - subset["feeder_billing_efficiency"]) +
                        w_dt_local * (1 - subset["dt_billing_efficiency"]) +
                        w_location_local * subset["location_trust_score"] +
                        p_n * subset["pattern_deviation_score"] +
                        r_n * subset["dt_relative_usage_score"] +
                        z_n * subset["zero_counter_score"]
                    )
                    mean_score = theft_scores.mean()
                    if mean_score > best_mean:
                        best_mean = mean_score
                        best_combo = (p_n, r_n, z_n)

        # compute subset post-optimized mean
        post_mean = best_mean
        return best_combo, pre_mean, post_mean
    except Exception as e:
        st.error(f"Optimizer failed: {e}")
        return (0.0,0.0,0.0), None, None

# Build customer_monthly_all theft probability using current sliders (so optimizer has baseline)
# For escalations we use customer_monthly_all (which contains all customers regardless of UI filters)
# Merge in pattern/zero/dt_relative/feeder/dt monthly metrics similar to above but using *_all variants
# First compute supportive tables for *_all

# Build dt_merged_monthly_all similar to dt_merged_monthly but using ALL customers aggregated
try:
    customer_billed_monthly_all = customer_monthly_all.groupby(["NAME_OF_DT","month"])["billed_kwh"].sum().reset_index().rename(columns={"billed_kwh":"customer_billed_kwh"})
    dt_merged_monthly_all = dt_agg_monthly.merge(customer_billed_monthly_all, left_on=["New Unique DT Nomenclature","month"], right_on=["NAME_OF_DT","month"], how="left")
    dt_merged_monthly_all["customer_billed_kwh"] = dt_merged_monthly_all["customer_billed_kwh"].fillna(0)
    dt_merged_monthly_all["total_billed_kwh"] = np.where(
        dt_merged_monthly_all["Ownership"].str.strip().str.upper().isin(["PRIVATE"]), dt_merged_monthly_all["total_dt_kwh"], dt_merged_monthly_all["customer_billed_kwh"]
    )
    dt_merged_monthly_all["dt_billing_efficiency"] = np.where(
        (dt_merged_monthly_all["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_merged_monthly_all["total_energy_kwh"] > 0),
        0.0,
        (dt_merged_monthly_all["total_billed_kwh"] / dt_merged_monthly_all["total_dt_kwh"].replace(0,1)).clip(0,1)
    )
    # attach location trust scores for dt and feeder for monthly all
    dt_merged_monthly_all = dt_merged_monthly_all.merge(dt_escalations[["DT Nomenclature","location_trust_score"]], left_on="New Unique DT Nomenclature", right_on="DT Nomenclature", how="left")
    dt_merged_monthly_all["location_trust_score"] = dt_merged_monthly_all["location_trust_score"].fillna(0)
except Exception as e:
    st.error(f"Building dt_merged_monthly_all failed: {e}")
    st.stop()

# feeder_merged_all
try:
    feeder_monthly_all = feeder_df.melt(id_vars=["Feeder","Feeder_Short","Tariff_Rate"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="feeder_energy_kwh")
    feeder_monthly_all["month"] = feeder_monthly_all["month"].str.replace(" (kWh)","")
    feeder_monthly_all["month"] = pd.Categorical(feeder_monthly_all["month"], categories=months, ordered=True)
    feeder_monthly_all = feeder_monthly_all[feeder_monthly_all["month"].isin(selected_months)]
    feeder_agg_all = feeder_monthly_all.groupby(["Feeder","Feeder_Short","Tariff_Rate"])["feeder_energy_kwh"].sum().reset_index()
    # total billed per feeder from dt_merged (use dt_agg_sum aggregated across dt_merged)
    # First compute dt_merged_all (sum over months)
    dt_agg_sum_all = dt_agg_monthly.groupby(["New Unique DT Nomenclature","DT_Short_Name","Feeder","Tariff_Rate","Ownership","Connection Status","total_energy_kwh"])["total_dt_kwh"].sum().reset_index()
    # To compute billed, we need customer aggregation over all customers (customer_monthly_all)
    cust_agg_all = customer_monthly_all.groupby(["NAME_OF_DT","Feeder"])["billed_kwh"].sum().reset_index().rename(columns={"billed_kwh":"customer_billed_kwh"})
    dt_merged_all = dt_agg_sum_all.merge(cust_agg_all, left_on=["New Unique DT Nomenclature","Feeder"], right_on=["NAME_OF_DT","Feeder"], how="left")
    dt_merged_all["customer_billed_kwh"] = dt_merged_all["customer_billed_kwh"].fillna(0)
    dt_merged_all["total_billed_kwh"] = np.where(
        dt_merged_all["Ownership"].str.strip().str.upper().isin(["PRIVATE"]), dt_merged_all["total_dt_kwh"], dt_merged_all["customer_billed_kwh"]
    )
    feeder_agg_billed_all = dt_merged_all.groupby("Feeder")["total_billed_kwh"].sum().reset_index()
    feeder_merged_all = feeder_agg_all.merge(feeder_agg_billed_all, on="Feeder", how="left")
    feeder_merged_all["total_billed_kwh"] = feeder_merged_all["total_billed_kwh"].fillna(0)
    feeder_merged_all["feeder_billing_efficiency"] = (feeder_merged_all["total_billed_kwh"] / feeder_merged_all["feeder_energy_kwh"].replace(0,1)).clip(0,1)
    feeder_merged_all = feeder_merged_all.merge(feeder_escalations[["Feeder","location_trust_score"]], on="Feeder", how="left")
    feeder_merged_all["location_trust_score"] = feeder_merged_all["location_trust_score"].fillna(0)
except Exception as e:
    st.error(f"Building feeder_merged_all failed: {e}")
    st.stop()

# Now annotate customer_monthly_all with feeder/dt monthly efficiencies & pattern/zero/relative scores
try:
    # compute pattern & zero & dt_relative for all customers (these functions expect one-row-per-customer shape)
    pattern_all_df = calculate_pattern_deviation(customer_df_all, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
    zero_all_df = calculate_zero_counter(customer_df_all, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
    # merge feeder metrics
    customer_monthly_all["energy_billed_score"] = (1 - customer_monthly_all["billed_kwh"] / customer_monthly_all["billed_kwh"].replace(0,1).max()).clip(0,1)
    # merge feeder metrics per feeder
    customer_monthly_all = customer_monthly_all.merge(feeder_merged_all[["Feeder","feeder_billing_efficiency","location_trust_score"]], on="Feeder", how="left")
    # merge dt monthly metrics
    customer_monthly_all = customer_monthly_all.merge(dt_merged_monthly_all[["New Unique DT Nomenclature","month","dt_billing_efficiency","location_trust_score"]], left_on=["NAME_OF_DT","month"], right_on=["New Unique DT Nomenclature","month"], how="left")
    # merge pattern and zero (one row per account)
    customer_monthly_all = customer_monthly_all.merge(pattern_all_df, left_on="ACCOUNT_NUMBER", right_on="id", how="left")
    customer_monthly_all = customer_monthly_all.merge(zero_all_df, left_on="ACCOUNT_NUMBER", right_on="id", how="left")
    # dt_relative_usage_all
    customer_monthly_all = customer_monthly_all.merge(dt_relative_usage_all, on="ACCOUNT_NUMBER", how="left")
    # fillna defaults
    customer_monthly_all["feeder_billing_efficiency"] = customer_monthly_all["feeder_billing_efficiency"].fillna(0)
    customer_monthly_all["dt_billing_efficiency"] = customer_monthly_all["dt_billing_efficiency"].fillna(0)
    customer_monthly_all["location_trust_score"] = customer_monthly_all["location_trust_score_x"].combine_first(customer_monthly_all["location_trust_score_y"]).fillna(0)
    customer_monthly_all["pattern_deviation_score"] = customer_monthly_all["pattern_deviation_score"].fillna(0)
    customer_monthly_all["zero_counter_score"] = customer_monthly_all["zero_counter_score"].fillna(0)
    customer_monthly_all["dt_relative_usage_score"] = customer_monthly_all["dt_relative_usage_score"].fillna(0)
    # compute theft_probability_all with current sliders
    customer_monthly_all["theft_probability"] = (
        w_feeder * (1 - customer_monthly_all["feeder_billing_efficiency"]) +
        w_dt * (1 - customer_monthly_all["dt_billing_efficiency"]) +
        w_location * customer_monthly_all["location_trust_score"] +
        w_pattern * customer_monthly_all["pattern_deviation_score"] +
        w_relative * customer_monthly_all["dt_relative_usage_score"] +
        w_zero * customer_monthly_all["zero_counter_score"]
    ).clip(0,1)
except Exception as e:
    st.error(f"Annotating customer_monthly_all failed: {e}")
    st.stop()

# -----------------------
# Optimizer UI & safe session_state update (no experimental_rerun)
# -----------------------
st.subheader("Optimizer")
if st.button("Optimize Customer-Level Weights for Escalations"):
    best_combo, pre_mean, post_mean = optimize_customer_weights(customer_monthly_all, escalations_df, w_feeder, w_dt, w_location)
    # update session_state safely
    st.session_state.update({
        "w_pattern": float(best_combo[0]),
        "w_relative": float(best_combo[1]),
        "w_zero": float(best_combo[2])
    })
    msg = f"Optimizer applied pattern={best_combo[0]:.3f}, relative={best_combo[1]:.3f}, zero={best_combo[2]:.3f}."
    if pre_mean is not None and post_mean is not None:
        msg += f" Escalation avg theft changed {pre_mean:.3f} → {post_mean:.3f}."
    st.success(msg)
    st.info("Sliders updated. Move a slider slightly or click any 'Generate' button to refresh computed displays.")

# -----------------------
# Produce Escalations Report (independent of UI filters) - new feature
# -----------------------
def generate_escalations_report(prepaid_df, postpaid_df, escalations_df_local, cust_monthly_all_df):
    """
    prepaid_df, postpaid_df: original raw customer PPM/PPD tables (before filter)
    escalations_df_local: escalations sheet (has 'Account No')
    cust_monthly_all_df: melted monthly customer data with theft_probability computed for each account-month
    Returns DataFrame with one row per matched customer-month (or one row per account with aggregated theft if you prefer).
    We'll produce one row per account containing customer info, feeder, DT, monthly readings JAN..JUN, and AvgTheftProbability.
    """
    results = []
    # combined customers for lookup
    lookup_df = pd.concat([prepaid_df, postpaid_df], ignore_index=True)
    # ensure ACCOUNT_NUMBER is string
    lookup_df["ACCOUNT_NUMBER"] = lookup_df["ACCOUNT_NUMBER"].astype(str)
    # build mapping of account -> aggregated monthly readings and theft probability from cust_monthly_all_df
    # aggregate per account across selected months
    agg_readings = cust_monthly_all_df.groupby("ACCOUNT_NUMBER").apply(
        lambda g: pd.Series({
            m: g[g["month"]==m]["billed_kwh"].sum() if m in g["month"].cat.categories else 0
            for m in cust_monthly_all_df["month"].cat.categories
        })
    ).reset_index().rename(columns={0:"_tmp"}).drop(columns=["_tmp"], errors="ignore")

    # simpler approach: pivot cust_monthly_all_df into account x months
    pivot = cust_monthly_all_df.pivot_table(index="ACCOUNT_NUMBER", columns="month", values="billed_kwh", aggfunc="sum").reset_index()
    # compute avg theft per account
    theft_avg = cust_monthly_all_df.groupby("ACCOUNT_NUMBER")["theft_probability"].mean().reset_index().rename(columns={"theft_probability":"Theft Probability (avg)"})
    # join pivot + theft_avg + customer info
    merged = pivot.merge(theft_avg, on="ACCOUNT_NUMBER", how="left")
    # left-join to lookup_df to grab customer metadata
    # choose columns to return: ACCOUNT_NUMBER, CUSTOMER_NAME, Feeder, NAME_OF_DT, monthly columns, Theft Probability
    lookup_meta = lookup_df[["ACCOUNT_NUMBER","CUSTOMER_NAME","NAME_OF_DT","NAME_OF_FEEDER","METER_NUMBER"]].drop_duplicates(subset=["ACCOUNT_NUMBER"])
    merged = merged.merge(lookup_meta, on="ACCOUNT_NUMBER", how="left")

    # Now iterate through escalations accounts and record
    for acc in escalations_df_local["Account No"].astype(str).tolist():
        rec = {"Account No": acc}
        row = merged[merged["ACCOUNT_NUMBER"].astype(str) == acc]
        if row.empty:
            # not found
            rec.update({
                "Customer Name": "Not Found",
                "Feeder": "Not Found",
                "DT": "Not Found",
                **{m: "" for m in months},
                "Theft Probability (avg)": ""
            })
        else:
            r = row.iloc[0]
            rec["Customer Name"] = r.get("CUSTOMER_NAME", "")
            # prefer NAME_OF_FEEDER if present else try to derive feeder from NAME_OF_DT
            feeder_val = r.get("NAME_OF_FEEDER", "")
            dt_val = r.get("NAME_OF_DT", "")
            rec["Feeder"] = feeder_val if pd.notna(feeder_val) else ""
            rec["DT"] = dt_val if pd.notna(dt_val) else ""
            # add months values (ensure numeric)
            for m in months:
                if m in r.index:
                    val = r[m] if pd.notna(r[m]) else 0
                    rec[m] = float(val)
                else:
                    rec[m] = 0.0
            # theft probability (may be NaN)
            tp = r.get("Theft Probability (avg)", "")
            rec["Theft Probability (avg)"] = float(tp) if (pd.notna(tp) and tp != "") else ""
        results.append(rec)

    report_df = pd.DataFrame(results)
    # order columns
    col_order = ["Account No", "Customer Name", "Feeder", "DT"] + months + ["Theft Probability (avg)"]
    # ensure all columns exist
    for c in col_order:
        if c not in report_df.columns:
            report_df[c] = ""
    return report_df[col_order]

# Button to generate escalations report and download
st.subheader("Escalations Report")
if st.button("Generate Escalations Report (checks all Escalations -> PPM/PPD)"):
    try:
        report_df = generate_escalations_report(ppm_df, ppd_df, escalations_df, customer_monthly_all)
        # styling: only format numeric theft probability
        styled_report = report_df.copy()
        if "Theft Probability (avg)" in styled_report.columns:
            styled_report["Theft Probability (avg)"] = pd.to_numeric(styled_report["Theft Probability (avg)"], errors="coerce")
            # convert floats -> show 3 decimals in style; keep other cols intact
            display_df = styled_report.copy()
            # use st.dataframe with style when numeric present
            styled = display_df.style.format({ "Theft Probability (avg)": "{:.3f}" })
            st.dataframe(styled, use_container_width=True)
        else:
            st.dataframe(styled_report.fillna(""), use_container_width=True)

        # Prepare excel for download
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            report_df.to_excel(writer, index=False, sheet_name="Escalations Report")
        buf.seek(0)
        st.download_button(label="📥 Download Escalations Report.xlsx", data=buf, file_name="Escalations_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Failed to generate escalations report: {e}")

# -----------------------
# The remainder of your original UI: feeder summary, DT summary, heatmaps, customer list, exports.
# I will reuse your original blocks but feeding the computed data frames above.
# -----------------------

# Feeder Summary
st.subheader("Feeder Summary")
if st.button("Show Feeder Summary"):
    try:
        required_cols = ["Feeder_Short", "feeder_energy_kwh", "total_billed_kwh", "feeder_energy_lost_kwh", "feeder_financial_loss_naira", "feeder_billing_efficiency", "location_trust_score"]
        missing_cols = [c for c in required_cols if c not in feeder_merged.columns]
        if missing_cols:
            st.error(f"Missing columns in feeder_merged: {missing_cols}")
        else:
            feeder_summary = feeder_merged[required_cols].copy()
            feeder_summary["Period"] = period_label
            feeder_summary.columns = ["Feeder", "Energy Supplied (kWh)", "Energy Billed (kWh)", "Energy Unaccounted For (kWh)", "Financial Loss (NGN)", "Billing Efficiency", "Location Trust Score", "Period"]
            feeder_summary = feeder_summary[["Feeder","Period","Energy Supplied (kWh)","Energy Billed (kWh)","Energy Unaccounted For (kWh)","Financial Loss (NGN)","Billing Efficiency","Location Trust Score"]]
            st.dataframe(feeder_summary.style.format({
                "Energy Supplied (kWh)":"{:.2f}",
                "Energy Billed (kWh)":"{:.2f}",
                "Energy Unaccounted For (kWh)":"{:.2f}",
                "Financial Loss (NGN)":"{:.2f}",
                "Billing Efficiency":"{:.3f}",
                "Location Trust Score":"{:.3f}"
            }), use_container_width=True)
    except Exception as e:
        st.error(f"Feeder summary failed: {e}")

# DT Summary
st.subheader(f"DT Summary for {selected_feeder_short}")
if selected_feeder_short and st.button("Show DT Summary"):
    try:
        if not selected_feeder:
            st.error("Selected feeder not found.")
        else:
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
            dt_summary = dt_summary[["DT","Period","Energy Supplied (kWh)","Energy Billed (kWh)","Energy Unaccounted For (kWh)","Financial Loss (NGN)","Billing Efficiency","Location Trust Score"]]
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

# DT Theft Probability Heatmap
st.subheader("DT Theft Probability Heatmap")
if selected_feeder:
    try:
        filtered_dt_agg = dt_merged_monthly[dt_merged_monthly["Feeder"] == selected_feeder]
        dt_theft_scores = filtered_dt_agg.groupby("DT_Short_Name")["dt_billing_efficiency"].mean().reset_index()
        dt_theft_scores["theft_probability"] = 1 - dt_theft_scores["dt_billing_efficiency"]
        dt_order = dt_theft_scores.sort_values("theft_probability", ascending=False)["DT_Short_Name"].tolist()
        if filtered_dt_agg.empty:
            st.error(f"No DT data for feeder {selected_feeder_short}.")
        else:
            dt_pivot = filtered_dt_agg.pivot_table(index="DT_Short_Name", columns="month", values="dt_billing_efficiency", aggfunc="mean").reindex(index=dt_order, columns=months)
            if not dt_pivot.empty:
                plt.figure(figsize=(10,8))
                sns.heatmap(1 - dt_pivot, cmap="YlOrRd", cbar_kws={"label":"DT Theft Probability"}, vmin=0, vmax=1)
                plt.title(f"DT Theft Probability for {selected_feeder_short} ({period_label}) (Ranked by Theft Probability)")
                st.pyplot(plt.gcf())
                plt.close()
            else:
                st.error(f"No DT data for {selected_feeder_short} after pivoting.")
    except Exception as e:
        st.error(f"DT heatmap failed: {e}")
else:
    st.warning("Select a feeder to view DT heatmap.")

# Customer scoring / heatmap / list (filtered)
st.subheader("Theft Analysis")
if selected_dt_short:
    try:
        filtered_customers = customer_monthly[customer_monthly["DT_Short_Name"] == selected_dt_short]
        if filtered_customers.empty:
            st.warning(f"No customer data for {selected_dt_short} ({period_label}).")
        else:
            customer_theft_scores = filtered_customers.groupby("ACCOUNT_NUMBER")["theft_probability"].mean().reset_index()
            customer_order = customer_theft_scores.sort_values("theft_probability", ascending=False)["ACCOUNT_NUMBER"].tolist()
            num_customers = st.number_input("Number of high-risk customers for Heatmap (0 for all)", min_value=0, value=min(10, len(customer_order)), step=1)
            if num_customers > 0:
                customer_subset = filtered_customers[filtered_customers["ACCOUNT_NUMBER"].isin(customer_order[:num_customers])]
            else:
                customer_subset = filtered_customers
            pivot_data = customer_subset.pivot_table(index="ACCOUNT_NUMBER", columns="month", values="theft_probability", aggfunc="mean").reindex(index=customer_order[:num_customers or None], columns=months)
            if not pivot_data.empty:
                plt.figure(figsize=(10,8))
                sns.heatmap(pivot_data, cmap="YlOrRd", vmin=0, vmax=1, cbar_kws={"label":"Theft Probability"})
                plt.title(f"Theft Probability for {selected_dt_short} ({selected_feeder_short}, {period_label}) (Ranked by Theft Probability)")
                st.pyplot(plt.gcf())
                plt.close()
            else:
                st.warning("No pivot data available for heatmap.")
    except Exception as e:
        st.error(f"Customer heatmap failed: {e}")
else:
    st.warning("Select a DT to view customer heatmap.")

# Customer List & export (filtered)
st.subheader(f"Customers under {selected_dt_short} ({selected_feeder_short}, {period_label})")
if selected_dt_short and st.button("Show Customer List"):
    try:
        filtered_customers = customer_monthly[customer_monthly["DT_Short_Name"] == selected_dt_short]
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
            missing_cols = [c for c in display_columns if c not in month_customers.columns]
            if missing_cols:
                st.error(f"Missing columns in month_customers: {missing_cols}")
            else:
                month_customers = month_customers.sort_values(by="theft_probability", ascending=False)
                styled_df = month_customers[display_columns].style.format({
                    "billed_kwh":"{:.2f}",
                    "theft_probability":"{:.3f}",
                    "pattern_deviation_score":"{:.3f}",
                    "dt_relative_usage_score":"{:.3f}",
                    "zero_counter_score":"{:.3f}"
                }).highlight_max(subset=["theft_probability"], color="lightcoral")
                st.dataframe(styled_df, use_container_width=True)
                # csv download
                csv = month_customers.to_csv(index=False)
                st.download_button(label=f"Download Customer List ({period_label})", data=csv, file_name=f"theft_analysis_{selected_dt_short}_{selected_feeder_short}_{period_label.replace(' ','_')}.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Customer list failed: {e}")

# Footer
st.markdown("Built by Elvis Ebenuwah for Ikeja Electric. 2025.")
