import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set Streamlit page config
st.set_page_config(page_title="IE Energy Theft Detection Dashboard", layout="wide")

# ------------------------
# Utility Functions
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
# New/Improved scoring functions
# ------------------------
def calculate_pattern_deviation(df, id_col, value_cols):
    """
    Pattern deviation: count months where reading < 60% of that customer's max (non-zero).
    If all months are zero -> maximum suspicion (score 1.0).
    Returns dataframe: id, pattern_deviation_score
    """
    results = []
    valid_value_cols = [c for c in value_cols if c in df.columns]
    if not valid_value_cols:
        return pd.DataFrame(columns=["id", "pattern_deviation_score"])
    for id_val, group in df.groupby(id_col):
        # We expect single row per account in the original customer tables
        values = group[valid_value_cols].iloc[0].values.astype(float)
        nonzero_values = values[values > 0]
        if len(nonzero_values) == 0:
            score = 1.0
        else:
            max_val = np.max(nonzero_values)
            below_threshold = np.sum(values < 0.6 * max_val)
            # use denominator = number of months (valid_value_cols)
            score = below_threshold / len(valid_value_cols)
        results.append({"id": id_val, "pattern_deviation_score": min(score, 1.0)})
    return pd.DataFrame(results)

def calculate_zero_counter(df, id_col, value_cols):
    """
    Zero counter: fraction of months with zero reading.
    """
    results = []
    valid_value_cols = [c for c in value_cols if c in df.columns]
    if not valid_value_cols:
        return pd.DataFrame(columns=["id", "zero_counter_score"])
    for id_val, group in df.groupby(id_col):
        values = group[valid_value_cols].iloc[0].values.astype(float)
        zeros = np.sum(values == 0)
        score = zeros / len(valid_value_cols)
        results.append({"id": id_val, "zero_counter_score": min(score, 1.0)})
    return pd.DataFrame(results)

def calculate_dt_relative_usage(customer_monthly):
    """
    Compute dt_relative_usage_score based on customer monthly billed_kwh aggregated across selected months.
    DT average excludes zero readings (per your request).
    Returns ACCOUNT_NUMBER -> dt_relative_usage_score
    """
    if customer_monthly.empty:
        return pd.DataFrame(columns=["ACCOUNT_NUMBER", "dt_relative_usage_score"])
    customer_agg = customer_monthly.groupby(["ACCOUNT_NUMBER", "NAME_OF_DT"])["billed_kwh"].sum().reset_index()
    # compute DT average ignoring zero monthly billed_kwh (we need per-customer sums; we will compute DT average over customers whose billed_kwh > 0)
    dt_avg = customer_monthly[customer_monthly["billed_kwh"] > 0].groupby("NAME_OF_DT")["billed_kwh"].sum().reset_index()
    # But the above dt_avg is sum-of-billed of customers with >0 across months; we need average per customer:
    # compute number of distinct customers per DT with >0
    dt_counts = customer_monthly[customer_monthly["billed_kwh"] > 0].groupby("NAME_OF_DT")["ACCOUNT_NUMBER"].nunique().reset_index().rename(columns={"ACCOUNT_NUMBER": "nonzero_customers"})
    dt_avg = dt_avg.merge(dt_counts, on="NAME_OF_DT", how="left")
    # Avoid division by zero
    dt_avg["dt_avg_kwh"] = np.where(dt_avg["nonzero_customers"] > 0, dt_avg["billed_kwh"] / dt_avg["nonzero_customers"], 0)
    dt_avg = dt_avg[["NAME_OF_DT", "dt_avg_kwh"]]
    customer_agg = customer_agg.merge(dt_avg, on="NAME_OF_DT", how="left")
    customer_agg["relative_ratio"] = np.where(customer_agg["dt_avg_kwh"] == 0, 0.5, customer_agg["billed_kwh"] / customer_agg["dt_avg_kwh"])
    # scoring formula as before, clipped
    def rel_score(row):
        if pd.isna(row["dt_avg_kwh"]) or row["dt_avg_kwh"] == 0:
            return 0.5  # neutral-ish when no DT average
        if row["billed_kwh"] < row["dt_avg_kwh"] * 0.3:
            return 0.9
        if row["billed_kwh"] > row["dt_avg_kwh"] * 0.7:
            return 0.1
        # linear interpolation
        return 0.1 + (0.9 - 0.1) * (0.7 - row["relative_ratio"]) / (0.7 - 0.3)
    customer_agg["dt_relative_usage_score"] = customer_agg.apply(lambda r: rel_score(r), axis=1).clip(0, 1)
    return customer_agg[["ACCOUNT_NUMBER", "dt_relative_usage_score"]]

# ------------------------
# Optimizer (searches pattern, relative, zero weights at step 0.05)
# ------------------------
def optimize_customer_weights_for_escalations(customer_monthly_all, escalations_accounts, w_feeder, w_dt, w_location, base_pattern, base_relative, base_zero):
    """
    Try combinations of pattern/relative/zero weights (grid 0..1 step 0.05, normalized among the three)
    Keep w_feeder, w_dt, w_location fixed.
    Returns best (pattern, relative, zero) normalized triple and (pre_mean, post_mean).
    """
    if customer_monthly_all.empty:
        return (base_pattern, base_relative, base_zero), None, None

    # subset to escalation accounts
    subset = customer_monthly_all[customer_monthly_all["ACCOUNT_NUMBER"].astype(str).isin(set(escalations_accounts))]
    if subset.empty:
        return (base_pattern, base_relative, base_zero), None, None

    # compute current mean (pre)
    pre = (
        w_feeder * (1 - subset["feeder_billing_efficiency"]) +
        w_dt * (1 - subset["dt_billing_efficiency"]) +
        w_location * subset["location_trust_score"] +
        base_pattern * subset["pattern_deviation_score"] +
        base_relative * subset["dt_relative_usage_score"] +
        base_zero * subset["zero_counter_score"]
    ).mean()

    best_mean = -np.inf
    best_combo = (base_pattern, base_relative, base_zero)

    vals = np.arange(0.0, 1.05, 0.05)
    for wp in vals:
        for wr in vals:
            for wz in vals:
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
    return best_combo, pre, best_mean

# ------------------------
# File upload & sheet read
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

# Access sheets
feeder_df = sheets.get("Feeder Data")
dt_df = sheets.get("Transformer Data")
ppm_df = sheets.get("Customer Data_PPM")
ppd_df = sheets.get("Customer Data_PPD")
band_df = sheets.get("Feeder Band")
tariff_df = sheets.get("Customer Tariffs")
escalations_df = sheets.get("Escalations")

# Basic sheet presence check
if any(df is None for df in [feeder_df, dt_df, ppm_df, ppd_df, band_df, tariff_df, escalations_df]):
    st.error("One or more sheets missing. Ensure the Excel file contains: Feeder Data, Transformer Data, Customer Data_PPM, Customer Data_PPD, Feeder Band, Customer Tariffs, Escalations.")
    st.stop()

# Validate column names (some permissive fixes)
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
            st.warning("Ownership column missing in Transformer Data. Assuming all DTs are public.")
            dt_df["Ownership"] = "PUBLIC"
        else:
            st.error(f"Missing columns in {name}: {missing_cols}")
            st.stop()

# Months config
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
month_indices = {m: i for i, m in enumerate(months)}

# Ensure month columns exist where expected
for df, name in [(ppm_df, "Customer Data_PPM"), (ppd_df, "Customer Data_PPD"), (feeder_df, "Feeder Data"), (dt_df, "Transformer Data")]:
    missing_months = [m for m in months if m not in df.columns]
    if missing_months:
        st.warning(f"Missing month columns in {name}: {missing_months}. Filling with 0.")
        for m in missing_months:
            df[m] = 0

# Handle missing non-month columns
for df, col in [(ppm_df, "TARIFF"), (ppd_df, "TARIFF"), (band_df, "BAND"), (tariff_df, "Tariff")]:
    if col not in df.columns:
        df[col] = ""
if "Short Name" not in band_df.columns:
    band_df["Short Name"] = band_df["Feeder"].apply(lambda x: get_short_name(x))

# Rate column detection
rate_col = next((col for col in ["Rate (NGN)", "Rate (₦)", "Rate", "RATE", "Rate(NGN)", "Rate(₦)"] if col in tariff_df.columns), None)
if rate_col:
    tariff_df["Rate (NGN)"] = pd.to_numeric(tariff_df[rate_col], errors="coerce").fillna(209.5)
else:
    tariff_df["Rate (NGN)"] = 209.5

# band-specific tariff rates
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

# Preprocess month kWh columns (units: feeder in MWh -> *1000; customers already in kWh)
for month in months:
    # feeder_df monthly -> convert unit *1000
    col = f"{month} (kWh)"
    if month in feeder_df.columns:
        feeder_df[col] = pd.to_numeric(feeder_df[month], errors="coerce").fillna(0) * 1000
    else:
        feeder_df[col] = 0
    # ppm and ppd
    for df in (ppm_df, ppd_df):
        if month in df.columns:
            df[f"{month} (kWh)"] = pd.to_numeric(df[month], errors="coerce").fillna(0)
        else:
            df[f"{month} (kWh)"] = 0
    # dt_df
    if month in dt_df.columns:
        dt_df[f"{month} (kWh)"] = pd.to_numeric(dt_df[month], errors="coerce").fillna(0)
    else:
        dt_df[f"{month} (kWh)"] = 0

# Drop original month columns to avoid confusion
for df in [feeder_df, dt_df, ppm_df, ppd_df]:
    df.drop(columns=[m for m in months if m in df.columns], errors="ignore", inplace=True)

# Filter NOT CONNECTED DTs with zero total energy
dt_df["total_energy_kwh"] = dt_df[[f"{m} (kWh)" for m in months]].sum(axis=1)
not_connected_zero = (dt_df["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_df["total_energy_kwh"] == 0)
excluded_dts = dt_df[not_connected_zero][["New Unique DT Nomenclature", "Connection Status", "total_energy_kwh"]]
if not excluded_dts.empty:
    st.warning(f"Excluding {len(excluded_dts)} DTs marked 'NOT CONNECTED' with zero energy across all months.")
dt_df = dt_df[~not_connected_zero]

# Normalize many name columns
to_normalize = [
    ("Feeder", feeder_df), ("NAME_OF_FEEDER", ppm_df), ("NAME_OF_FEEDER", ppd_df), ("Feeder", band_df),
    ("NAME_OF_DT", ppm_df), ("NAME_OF_DT", ppd_df), ("New Unique DT Nomenclature", dt_df),
    ("TARIFF", ppm_df), ("TARIFF", ppd_df), ("Tariff", tariff_df),
    ("BUSINESS_UNIT", ppm_df), ("BUSINESS_UNIT", ppd_df),
    ("UNDERTAKING", ppm_df), ("UNDERTAKING", ppd_df),
    ("Feeder", escalations_df), ("DT Nomenclature", escalations_df)
]
for col, df in to_normalize:
    if col in df.columns:
        df[col] = df[col].apply(normalize_name)

# Combine PPM + PPD into customer_df (this will be filtered later)
ppm_df["Billing_Type"] = "PPM"
ppd_df["Billing_Type"] = "PPD"
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True, sort=False)
if customer_df.empty:
    st.error("customer_df is empty.")
    st.stop()

# Keep a full customer lookup (this is used for Escalations independent lookup)
customer_lookup_df = customer_df.copy()  # before later filtering by valid DT/Feeder
# Ensure a Feeder column exists on lookup dataset (derive from NAME_OF_DT)
def derive_feeder_from_dt(x):
    if isinstance(x, str) and "-" in x and len(x.split("-")) >= 3:
        return "-".join(x.split("-")[:-1])
    return x
customer_lookup_df["Feeder"] = customer_lookup_df["NAME_OF_DT"].apply(derive_feeder_from_dt).apply(normalize_name)

# Filter for valid DTs (as per original app behavior) and build working customer_df
valid_dts = set(dt_df["New Unique DT Nomenclature"])
customer_invalid_dts = customer_df[~customer_df["NAME_OF_DT"].isin(valid_dts)]
error_report = []
if not customer_invalid_dts.empty:
    for _, row in customer_invalid_dts.iterrows():
        error_report.append({
            "ACCOUNT_NUMBER": row.get("ACCOUNT_NUMBER", ""),
            "NAME_OF_DT": row.get("NAME_OF_DT", ""),
            "NAME_OF_FEEDER": row.get("NAME_OF_FEEDER", ""),
            "BUSINESS_UNIT": row.get("BUSINESS_UNIT", ""),
            "UNDERTAKING": row.get("UNDERTAKING", ""),
            "Reason": "NAME_OF_DT not in Transformer Data"
        })
error_report_df = pd.DataFrame(error_report)
customer_df = customer_df[customer_df["NAME_OF_DT"].isin(valid_dts)]
if customer_df.empty:
    st.error("No valid customers after filtering for Transformer Data DTs.")
    st.stop()

# Create short names and feeder values for dt_df and customer_df
feeder_df["Feeder_Short"] = feeder_df["Feeder"].apply(lambda x: get_short_name(x))
dt_df["DT_Short_Name"] = dt_df["New Unique DT Nomenclature"].apply(lambda x: get_short_name(x, is_dt=True))
dt_df["Feeder"] = dt_df["New Unique DT Nomenclature"].apply(lambda x: derive_feeder_from_dt(x))
dt_df["Feeder"] = dt_df["Feeder"].apply(normalize_name)
dt_df["NAME_OF_DT"] = dt_df["New Unique DT Nomenclature"]

customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))
customer_df["Feeder"] = customer_df["NAME_OF_DT"].apply(lambda x: derive_feeder_from_dt(x))
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

# Merge tariffs into customer_df
tariff_matches = customer_df["TARIFF"].isin(tariff_df["Tariff"])
if not tariff_matches.all():
    st.warning(f"Some TARIFF values in customer data not found in Customer Tariffs: {customer_df[~tariff_matches]['TARIFF'].unique()}")
customer_df = customer_df.merge(tariff_df[["Tariff", "Rate (NGN)"]], left_on="TARIFF", right_on="Tariff", how="left")
customer_df["Rate (NGN)"] = customer_df["Rate (NGN)"].fillna(209.5)
customer_df = customer_df.drop(columns=["Tariff"], errors="ignore")

# Calculate location trust scores from Escalations (for feeders & DTs)
escalations_df["Report_Count"] = 1
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

# ------------------------
# Streamlit UI Filters & Weight sliders
# ------------------------
st.title("Ikeja Electric Energy Theft Detection Dashboard")
st.subheader("Filters")
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    bu_options = sorted(customer_df["BUSINESS_UNIT"].dropna().unique())
    selected_bu = st.selectbox("Select Business Unit", bu_options) if bu_options else ""
    if not bu_options:
        st.warning("No Business Units available.")
with col2:
    if selected_bu:
        customer_df_bu = customer_df[customer_df["BUSINESS_UNIT"] == selected_bu]
        ut_options = sorted(customer_df_bu["UNDERTAKING"].dropna().unique())
        selected_ut = st.selectbox("Select Undertaking", ut_options) if ut_options else ""
        if not ut_options:
            st.warning("No Undertakings available for selected BU.")
    else:
        customer_df_bu = pd.DataFrame()
        selected_ut = ""
with col3:
    if selected_ut:
        customer_df_ut = customer_df_bu[customer_df_bu["UNDERTAKING"] == selected_ut]
        feeder_options = sorted(feeder_df["Feeder_Short"].dropna().unique())
        selected_feeder_short = st.selectbox("Select Feeder", feeder_options) if feeder_options else ""
        if not feeder_options:
            st.error("No feeders available in Feeder Data.")
            st.stop()
    else:
        customer_df_ut = pd.DataFrame()
        selected_feeder_short = ""
with col4:
    if selected_feeder_short:
        selected_feeder = feeder_df[feeder_df["Feeder_Short"] == selected_feeder_short]["Feeder"].iloc[0]
        dt_df_filtered = dt_df[dt_df["Feeder"] == selected_feeder]
        dt_options = sorted(dt_df_filtered["DT_Short_Name"].dropna().unique())
        selected_dt_short = st.selectbox("Select DT", dt_options) if dt_options else ""
        if selected_dt_short == "":
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

# Dynamic Weight Sliders (include zero-frequency)
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

# Normalize weights (all weights together)
total_weight = w_feeder + w_dt + w_location + w_pattern + w_relative + w_zero
if total_weight == 0:
    st.error("Total weight cannot be zero. Please adjust weights.")
    st.stop()

w_feeder /= total_weight
w_dt /= total_weight
w_location /= total_weight
w_pattern /= total_weight
w_relative /= total_weight
w_zero /= total_weight

# ------------------------
# Use filtered customer_df_ut if selected, else full customer_df
# ------------------------
if 'customer_df_ut' not in locals() or customer_df_ut.empty:
    st.warning("No filtered customer data available. Using full dataset.")
    customer_df_filtered = customer_df.copy()
else:
    customer_df_filtered = customer_df_ut.copy()

customer_df = customer_df_filtered.copy()
dt_df = dt_df[dt_df["Feeder"].isin(valid_feeders)]
if dt_df.empty or customer_df.empty:
    st.error("No valid data after BU/UT filtering.")
    st.stop()

# ------------------------
# Calculate pattern/zero for current (filtered) customers (used in UI analysis)
# ------------------------
# compute pattern and zero using customer_df (row-per-customer)
pattern_df = calculate_pattern_deviation(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
zero_df = calculate_zero_counter(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])

# ------------------------
# Melt customer data into monthly rows (filtered)
# ------------------------
required_id_vars = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type", "Feeder", "Rate (NGN)"]
# ensure required id vars exist in customer_df
for c in required_id_vars:
    if c not in customer_df.columns:
        customer_df[c] = ""

value_vars = [f"{m} (kWh)" for m in months]
try:
    customer_monthly = customer_df.melt(id_vars=required_id_vars, value_vars=value_vars, var_name="month", value_name="billed_kwh")
    customer_monthly["month"] = customer_monthly["month"].str.replace(" (kWh)", "")
    customer_monthly["month"] = pd.Categorical(customer_monthly["month"], categories=months, ordered=True)
except Exception as e:
    st.error(f"Melt failed for customer data: {e}")
    st.stop()

# ------------------------
# DT monthly melt for heatmaps
# ------------------------
try:
    dt_agg_monthly = dt_df.melt(id_vars=["New Unique DT Nomenclature", "DT_Short_Name", "Feeder", "Tariff_Rate", "Ownership", "Connection Status", "total_energy_kwh"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="total_dt_kwh")
    dt_agg_monthly["month"] = dt_agg_monthly["month"].str.replace(" (kWh)", "")
    dt_agg_monthly["month"] = pd.Categorical(dt_agg_monthly["month"], categories=months, ordered=True)
except Exception as e:
    st.error(f"DT melt failed: {e}")
    st.stop()

# Filter by user-selected month range
selected_months = months[month_indices[start_month]:month_indices[end_month] + 1]
if not selected_months:
    st.error("No months selected.")
    st.stop()
customer_monthly = customer_monthly[customer_monthly["month"].isin(selected_months)]
dt_agg_monthly = dt_agg_monthly[dt_agg_monthly["month"].isin(selected_months)]
if customer_monthly.empty or dt_agg_monthly.empty:
    st.error(f"No data for selected months {selected_months}.")
    st.stop()

# ------------------------
# DT relative usage (filtered customers)
# ------------------------
dt_relative_usage = calculate_dt_relative_usage(customer_monthly)

# ------------------------
# Aggregations for DT and Feeder metrics
# ------------------------
period_label = f"{start_month}" if start_month == end_month else f"{start_month} to {end_month}"
try:
    customer_agg = customer_monthly.groupby(["NAME_OF_DT", "Feeder"])["billed_kwh"].sum().reset_index().rename(columns={"billed_kwh": "customer_billed_kwh"})
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
    dt_merged = dt_agg_sum.merge(customer_agg, left_on=["New Unique DT Nomenclature", "Feeder"], right_on=["NAME_OF_DT", "Feeder"], how="left")
    dt_merged["customer_billed_kwh"] = dt_merged["customer_billed_kwh"].fillna(0)
    dt_merged["total_billed_kwh"] = np.where(dt_merged["Ownership"].str.strip().str.upper().isin(["PRIVATE"]), dt_merged["total_dt_kwh"], dt_merged["customer_billed_kwh"])
    dt_merged["dt_billing_efficiency"] = np.where((dt_merged["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_merged["total_energy_kwh"] > 0), 0.0, (dt_merged["total_billed_kwh"] / dt_merged["total_dt_kwh"].replace(0, 1)).clip(0, 1))
    dt_merged["energy_lost_kwh"] = dt_merged["total_dt_kwh"] - dt_merged["total_billed_kwh"]
    dt_merged["financial_loss_naira"] = dt_merged["energy_lost_kwh"] * dt_merged["Tariff_Rate"]
except Exception as e:
    st.error(f"DT merge failed: {e}")
    st.stop()

# Per-month DT billing efficiency for heatmap (filtered)
try:
    customer_billed_monthly = customer_monthly.groupby(["NAME_OF_DT", "month"])["billed_kwh"].sum().reset_index().rename(columns={"billed_kwh": "customer_billed_kwh"})
    dt_merged_monthly = dt_agg_monthly.merge(customer_billed_monthly, left_on=["New Unique DT Nomenclature", "month"], right_on=["NAME_OF_DT", "month"], how="left")
    dt_merged_monthly["customer_billed_kwh"] = dt_merged_monthly["customer_billed_kwh"].fillna(0)
    dt_merged_monthly["total_billed_kwh"] = np.where(dt_merged_monthly["Ownership"].str.strip().str.upper().isin(["PRIVATE"]), dt_merged_monthly["total_dt_kwh"], dt_merged_monthly["customer_billed_kwh"])
    dt_merged_monthly["dt_billing_efficiency"] = np.where((dt_merged_monthly["Connection Status"].str.strip().str.upper() == "NOT CONNECTED") & (dt_merged_monthly["total_energy_kwh"] > 0), 0.0, (dt_merged_monthly["total_billed_kwh"] / dt_merged_monthly["total_dt_kwh"].replace(0, 1)).clip(0, 1))
except Exception as e:
    st.error(f"DT monthly merge failed: {e}")
    st.stop()

# Feeder consumption & billing efficiency (filtered)
try:
    feeder_monthly = feeder_df.melt(id_vars=["Feeder", "Feeder_Short", "Tariff_Rate"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="feeder_energy_kwh")
    feeder_monthly["month"] = feeder_monthly["month"].str.replace(" (kWh)", "")
    feeder_monthly["month"] = pd.Categorical(feeder_monthly["month"], categories=months, ordered=True)
    feeder_monthly = feeder_monthly[feeder_monthly["month"].isin(selected_months)]
    feeder_agg = feeder_monthly.groupby(["Feeder", "Feeder_Short", "Tariff_Rate"])["feeder_energy_kwh"].sum().reset_index()
    feeder_agg_billed = dt_merged.groupby(["Feeder"])["total_billed_kwh"].sum().reset_index()
    feeder_merged = feeder_agg.merge(feeder_agg_billed, on=["Feeder"], how="left")
    feeder_merged["total_billed_kwh"] = feeder_merged["total_billed_kwh"].fillna(0)
    feeder_merged["feeder_billing_efficiency"] = (feeder_merged["total_billed_kwh"] / feeder_merged["feeder_energy_kwh"].replace(0, 1)).clip(0, 1)
    feeder_merged["feeder_energy_lost_kwh"] = feeder_merged["feeder_energy_kwh"] - feeder_merged["total_billed_kwh"]
    feeder_merged["feeder_financial_loss_naira"] = feeder_merged["feeder_energy_lost_kwh"] * feeder_merged["Tariff_Rate"]
except Exception as e:
    st.error(f"Feeder melt/merge failed: {e}")
    st.stop()

# Merge location trust scores
feeder_merged = feeder_merged.merge(feeder_escalations[["Feeder", "location_trust_score"]], on="Feeder", how="left")
feeder_merged["location_trust_score"] = feeder_merged["location_trust_score"].fillna(0)
dt_merged = dt_merged.merge(dt_escalations[["DT Nomenclature", "location_trust_score"]], left_on="New Unique DT Nomenclature", right_on="DT Nomenclature", how="left")
dt_merged["location_trust_score"] = dt_merged["location_trust_score"].fillna(0)
dt_merged_monthly = dt_merged_monthly.merge(dt_escalations[["DT Nomenclature", "location_trust_score"]], left_on="New Unique DT Nomenclature", right_on="DT Nomenclature", how="left")
dt_merged_monthly["location_trust_score"] = dt_merged_monthly["location_trust_score"].fillna(0)

# ------------------------
# Customer scoring for filtered dataset (per-month)
# ------------------------
try:
    if customer_monthly.empty:
        st.error("No customer monthly data available.")
        st.stop()
    # simple normalization for billed score (used earlier)
    customer_monthly["energy_billed_score"] = (1 - customer_monthly["billed_kwh"] / customer_monthly["billed_kwh"].replace(0, 1).max()).clip(0, 1)
    # merge feeder metrics
    customer_monthly = customer_monthly.merge(feeder_merged[["Feeder", "feeder_billing_efficiency", "location_trust_score"]], on="Feeder", how="left")
    # merge DT monthly metrics
    customer_monthly = customer_monthly.merge(dt_merged_monthly[["New Unique DT Nomenclature", "month", "dt_billing_efficiency", "location_trust_score"]], left_on=["NAME_OF_DT", "month"], right_on=["New Unique DT Nomenclature", "month"], how="left")
    # merge pattern & zero & relative (pattern/zero computed at customer_df level earlier; relative is computed per-month)
    customer_monthly = customer_monthly.merge(pattern_df, left_on=["ACCOUNT_NUMBER"], right_on=["id"], how="left")
    customer_monthly = customer_monthly.merge(zero_df, left_on=["ACCOUNT_NUMBER"], right_on=["id"], how="left")
    customer_monthly = customer_monthly.merge(dt_relative_usage, on="ACCOUNT_NUMBER", how="left")
    customer_monthly["feeder_billing_efficiency"] = customer_monthly["feeder_billing_efficiency"].fillna(0)
    customer_monthly["dt_billing_efficiency"] = customer_monthly["dt_billing_efficiency"].fillna(0)
    customer_monthly["location_trust_score"] = customer_monthly["location_trust_score_x"].combine_first(customer_monthly["location_trust_score_y"]).fillna(0)
    customer_monthly["pattern_deviation_score"] = customer_monthly["pattern_deviation_score"].fillna(0)
    customer_monthly["zero_counter_score"] = customer_monthly["zero_counter_score"].fillna(0)
    customer_monthly["dt_relative_usage_score"] = customer_monthly["dt_relative_usage_score"].fillna(0)
    customer_monthly["theft_probability"] = (
        w_feeder * (1 - customer_monthly["feeder_billing_efficiency"]) +
        w_dt * (1 - customer_monthly["dt_billing_efficiency"]) +
        w_location * customer_monthly["location_trust_score"] +
        w_pattern * customer_monthly["pattern_deviation_score"] +
        w_relative * customer_monthly["dt_relative_usage_score"] +
        w_zero * customer_monthly["zero_counter_score"]
    ).clip(0, 1)
    customer_monthly["risk_tier"] = pd.cut(customer_monthly["theft_probability"], bins=[0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"], include_lowest=True)
except Exception as e:
    st.error(f"Customer scoring failed: {e}")
    st.stop()

# ------------------------
# Build full customer_monthly_all from customer_lookup_df (used for Escalations report and optimizer)
# ------------------------
# Ensure lookup has tariff Rate (merge)
customer_lookup_df = customer_lookup_df.merge(tariff_df[["Tariff", "Rate (NGN)"]], left_on="TARIFF", right_on="Tariff", how="left")
customer_lookup_df["Rate (NGN)"] = customer_lookup_df["Rate (NGN)"].fillna(209.5)
customer_lookup_df = customer_lookup_df.drop(columns=["Tariff"], errors="ignore")

# Ensure required id vars exist on lookup
for c in ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type", "Feeder", "Rate (NGN)"]:
    if c not in customer_lookup_df.columns:
        customer_lookup_df[c] = ""

# Create DT_Short_Name if not present
if "DT_Short_Name" not in customer_lookup_df.columns or customer_lookup_df["DT_Short_Name"].isnull().all():
    customer_lookup_df["DT_Short_Name"] = customer_lookup_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))

# Melt lookup to monthly rows
try:
    customer_monthly_all = customer_lookup_df.melt(
        id_vars=["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type", "Feeder", "Rate (NGN)"],
        value_vars=[f"{m} (kWh)" for m in months],
        var_name="month",
        value_name="billed_kwh"
    )
    customer_monthly_all["month"] = customer_monthly_all["month"].str.replace(" (kWh)", "")
    customer_monthly_all["month"] = pd.Categorical(customer_monthly_all["month"], categories=months, ordered=True)
    # filter to same selected months to be consistent with UI-calculated theft scores
    customer_monthly_all = customer_monthly_all[customer_monthly_all["month"].isin(selected_months)]
    customer_monthly_all["billed_kwh"] = pd.to_numeric(customer_monthly_all["billed_kwh"], errors="coerce").fillna(0)
except Exception as e:
    st.error(f"Failed to create full customer monthly dataset for escalations: {e}")
    st.stop()

# Merge feeder & dt metrics into the full customer monthly
# feeder metrics: feeder_merged has feeder_billing_efficiency + location_trust_score
customer_monthly_all = customer_monthly_all.merge(feeder_merged[["Feeder", "feeder_billing_efficiency", "location_trust_score"]], on="Feeder", how="left")
customer_monthly_all = customer_monthly_all.merge(dt_merged_monthly[["New Unique DT Nomenclature", "month", "dt_billing_efficiency", "location_trust_score"]], left_on=["NAME_OF_DT", "month"], right_on=["New Unique DT Nomenclature", "month"], how="left")
customer_monthly_all["feeder_billing_efficiency"] = customer_monthly_all["feeder_billing_efficiency"].fillna(0)
customer_monthly_all["dt_billing_efficiency"] = customer_monthly_all["dt_billing_efficiency"].fillna(0)
# combine location trust score (feeder-level may differ from dt-level)
customer_monthly_all["location_trust_score"] = customer_monthly_all["location_trust_score_x"].combine_first(customer_monthly_all["location_trust_score_y"]).fillna(0)
customer_monthly_all.drop(columns=[c for c in ["location_trust_score_x", "location_trust_score_y", "New Unique DT Nomenclature"] if c in customer_monthly_all.columns], errors="ignore", inplace=True)

# Compute per-customer pattern/zero on lookup (row-per-customer)
pattern_df_all = calculate_pattern_deviation(customer_lookup_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
zero_df_all = calculate_zero_counter(customer_lookup_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
# dt_relative for all customers (aggregated across selected months)
dt_relative_all = calculate_dt_relative_usage(customer_monthly_all)

# merge these into monthly all
customer_monthly_all = customer_monthly_all.merge(pattern_df_all, left_on="ACCOUNT_NUMBER", right_on="id", how="left")
customer_monthly_all = customer_monthly_all.merge(zero_df_all, left_on="ACCOUNT_NUMBER", right_on="id", how="left")
customer_monthly_all = customer_monthly_all.merge(dt_relative_all, on="ACCOUNT_NUMBER", how="left")
customer_monthly_all[["pattern_deviation_score", "zero_counter_score", "dt_relative_usage_score"]] = customer_monthly_all[["pattern_deviation_score", "zero_counter_score", "dt_relative_usage_score"]].fillna(0)

# Compute theft probability for full dataset (uses current slider weights)
customer_monthly_all["theft_probability"] = (
    w_feeder * (1 - customer_monthly_all["feeder_billing_efficiency"]) +
    w_dt * (1 - customer_monthly_all["dt_billing_efficiency"]) +
    w_location * customer_monthly_all["location_trust_score"] +
    w_pattern * customer_monthly_all["pattern_deviation_score"] +
    w_relative * customer_monthly_all["dt_relative_usage_score"] +
    w_zero * customer_monthly_all["zero_counter_score"]
).clip(0, 1)

# Aggregate per-account theft (average across months)
theft_by_account = customer_monthly_all.groupby("ACCOUNT_NUMBER")["theft_probability"].mean().reset_index().rename(columns={"theft_probability": "theft_probability_mean"})
theft_by_account["risk_tier"] = pd.cut(theft_by_account["theft_probability_mean"], bins=[0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"], include_lowest=True)

# ------------------------
# Optimizer button (search pattern/relative/zero values step 0.05)
# ------------------------
if st.button("Optimize Customer-Level Weights for Escalations"):
    escalation_accounts = escalations_df["Account No"].astype(str).tolist()
    base_pattern = st.session_state.get("w_pattern", w_pattern)
    base_relative = st.session_state.get("w_relative", w_relative)
    base_zero = st.session_state.get("w_zero", w_zero)
    best_combo, pre_mean, post_mean = optimize_customer_weights_for_escalations(customer_monthly_all, escalation_accounts, w_feeder, w_dt, w_location, base_pattern, base_relative, base_zero)
    st.session_state.update({
    "w_pattern": float(wp_n),
    "w_relative": float(wr_n),
    "w_zero": float(wz_n),
})
msg = f"Optimizer applied pattern={wp_n:.3f}, relative={wr_n:.3f}, zero={wz_n:.3f}."
if pre_mean is not None and post_mean is not None:
    msg += f" Escalation avg theft improved {pre_mean:.3f} → {post_mean:.3f}."
st.success(msg)
st.info("Sliders updated — move them slightly or regenerate to see new theft scores.")

    if best_combo is not None:
        # apply best combo to session state and rerun so sliders show updated values
        st.session_state.w_pattern = float(best_combo[0])
        st.session_state.w_relative = float(best_combo[1])
        st.session_state.w_zero = float(best_combo[2])
        msg = f"Optimizer applied pattern={best_combo[0]:.3f}, relative={best_combo[1]:.3f}, zero={best_combo[2]:.3f}."
        if pre_mean is not None and post_mean is not None:
            msg += f" Escalation avg theft improved {pre_mean:.3f} → {post_mean:.3f}."
        st.success(msg)
    else:
        st.info("Optimizer found no improvement or no escalation accounts present.")

# ------------------------
# UI visuals (Feeder summary / DT summary / heatmaps / customer list) - unchanged behavior
# (I kept these blocks mostly as in your original code)
# ------------------------

# Feeder Summary
st.subheader("Feeder Summary")
if st.button("Show Feeder Summary"):
    try:
        required_cols = ["Feeder_Short", "feeder_energy_kwh", "total_billed_kwh", "feeder_energy_lost_kwh", "feeder_financial_loss_naira", "feeder_billing_efficiency", "location_trust_score"]
        missing_cols = [col for col in required_cols if col not in feeder_merged.columns]
        if missing_cols:
            st.error(f"Missing columns in feeder_merged: {missing_cols}")
            st.stop()
        feeder_summary = feeder_merged[required_cols].copy()
        feeder_summary["Period"] = period_label
        feeder_summary.columns = ["Feeder", "Energy Supplied (kWh)", "Energy Billed (kWh)", "Energy Unaccounted For (kWh)", "Financial Loss (NGN)", "Billing Efficiency", "Location Trust Score", "Period"]
        feeder_summary = feeder_summary[["Feeder", "Period", "Energy Supplied (kWh)", "Energy Billed (kWh)", "Energy Unaccounted For (kWh)", "Financial Loss (NGN)", "Billing Efficiency", "Location Trust Score"]]
        st.dataframe(feeder_summary.style.format({
            "Energy Supplied (kWh)": "{:.2f}",
            "Energy Billed (kWh)": "{:.2f}",
            "Energy Unaccounted For (kWh)": "{:.2f}",
            "Financial Loss (NGN)": "{:.2f}",
            "Billing Efficiency": "{:.3f}",
            "Location Trust Score": "{:.3f}"
        }))
    except Exception as e:
        st.error(f"Feeder summary failed: {e}")

# DT Summary Table
st.subheader(f"DT Summary for {selected_feeder_short}")
if selected_feeder_short and st.button("Show DT Summary"):
    try:
        if not selected_feeder:
            st.error("Selected feeder not found.")
            st.stop()
        dt_summary = dt_merged[dt_merged["Feeder"] == selected_feeder].groupby(["DT_Short_Name"]).agg({
            "total_dt_kwh": "sum",
            "total_billed_kwh": "sum",
            "energy_lost_kwh": "sum",
            "financial_loss_naira": "sum",
            "dt_billing_efficiency": "mean",
            "location_trust_score": "mean"
        }).reset_index()
        dt_summary["Period"] = period_label
        dt_summary.columns = ["DT", "Energy Supplied (kWh)", "Energy Billed (kWh)", "Energy Unaccounted For (kWh)", "Financial Loss (NGN)", "Billing Efficiency", "Location Trust Score", "Period"]
        dt_summary = dt_summary[["DT", "Period", "Energy Supplied (kWh)", "Energy Billed (kWh)", "Energy Unaccounted For (kWh)", "Financial Loss (NGN)", "Billing Efficiency", "Location Trust Score"]]
        st.dataframe(dt_summary.style.format({
            "Energy Supplied (kWh)": "{:.2f}",
            "Energy Billed (kWh)": "{:.2f}",
            "Energy Unaccounted For (kWh)": "{:.2f}",
            "Financial Loss (NGN)": "{:.2f}",
            "Billing Efficiency": "{:.3f}",
            "Location Trust Score": "{:.3f}"
        }))
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
                plt.figure(figsize=(10, 8))
                sns.heatmap(1 - dt_pivot, cmap="YlOrRd", cbar_kws={"label": "DT Theft Probability"}, vmin=0, vmax=1)
                plt.title(f"DT Theft Probability for {selected_feeder_short} ({period_label}) (Ranked by Theft Probability)")
                st.pyplot(plt.gcf())
                plt.close()
            else:
                st.error(f"No DT data for {selected_feeder_short} after pivoting.")
    except Exception as e:
        st.error(f"DT heatmap failed: {e}")
else:
    st.warning("Select a feeder to view DT heatmap.")

# Customer Heatmap
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
                filtered_customers_heatmap = filtered_customers[filtered_customers["ACCOUNT_NUMBER"].isin(customer_order[:num_customers])]
            else:
                filtered_customers_heatmap = filtered_customers
            pivot_data = filtered_customers_heatmap.pivot_table(index="ACCOUNT_NUMBER", columns="month", values="theft_probability", aggfunc="mean").reindex(index=customer_order[:num_customers or None], columns=months)
            if not pivot_data.empty:
                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot_data, cmap="YlOrRd", vmin=0, vmax=1, cbar_kws={"label": "Theft Probability"})
                plt.title(f"Theft Probability for {selected_dt_short} ({selected_feeder_short}, {period_label}) (Ranked by Theft Probability)")
                st.pyplot(plt.gcf())
                plt.close()
            else:
                st.warning(f"No pivot data available for heatmap.")
    except Exception as e:
        st.error(f"Customer heatmap failed: {e}")
else:
    st.warning("Select a DT to view customer heatmap.")

# Customer List (filtered)
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
                "risk_tier": lambda x: pd.Series(x).mode()[0] if not x.mode().empty else "Unknown",
                "pattern_deviation_score": "mean",
                "dt_relative_usage_score": "mean",
                "zero_counter_score": "mean"
            }).reset_index()
            display_columns = ["ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "billed_kwh", "Billing_Type", "theft_probability", "risk_tier", "pattern_deviation_score", "dt_relative_usage_score", "zero_counter_score"]
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

# CSV Export for filtered customer list
st.subheader("Export Customer Data (filtered)")
if selected_dt_short and 'month_customers' in locals() and not month_customers.empty:
    csv = month_customers.to_csv(index=False)
    st.download_button(label=f"Download Customer List ({period_label})", data=csv, file_name=f"theft_analysis_{selected_dt_short}_{selected_feeder_short}_{period_label.replace(' ', '_')}.csv", mime="text/csv")
else:
    st.info("Generate the customer list first to enable export.")

# ------------------------
# Escalations Report (independent of filters) and download
# ------------------------
def generate_escalations_report(prepaid_df, postpaid_df, escalations_df, customer_monthly_all, theft_by_account):
    """
    For each Account No in escalations_df['Account No']:
      - look up account in prepaid_df or postpaid_df (original customer lists)
      - if found, return customer details + monthly readings (for selected months) + theft score (mean)
      - if not found, mark Not Found
    Note: prepaid_df/postpaid_df expected to have columns like ACCOUNT_NUMBER, CUSTOMER_NAME, NAME_OF_DT, METER_NUMBER,
          and monthly columns like 'JAN (kWh)' ... 'JUN (kWh)'
    """
    results = []
    # build a combined lookup (row-per-customer) from prepaid and postpaid as they were read
    lookup = pd.concat([prepaid_df, postpaid_df], ignore_index=True, sort=False)
    # ensure derived feeder column on lookup
    if "Feeder" not in lookup.columns:
        lookup["Feeder"] = lookup["NAME_OF_DT"].apply(lambda x: derive_feeder_from_dt(x) if isinstance(x, str) else "")
        lookup["Feeder"] = lookup["Feeder"].apply(normalize_name)

    # theft_by_account is DataFrame ACCOUNT_NUMBER -> theft_probability_mean
    theft_map = theft_by_account.set_index("ACCOUNT_NUMBER")["theft_probability_mean"].to_dict()

    for acc in escalations_df["Account No"].astype(str).tolist():
        row_lookup = lookup[lookup["ACCOUNT_NUMBER"].astype(str) == str(acc)]
        if row_lookup.empty:
            # not found
            results.append({
                "Account No": acc,
                "Found": False,
                "Customer Name": "",
                "Billing_Type": "",
                "Feeder": "",
                "DT": "",
                "METER_NUMBER": "",
                **{m: None for m in months},
                "Theft Probability (avg)": None,
                "Risk Tier": ""
            })
        else:
            # one account may have multiple rows (unlikely) - include all matches
            for _, r in row_lookup.iterrows():
                # fetch monthly readings from the lookup row (we created monthly kWh columns earlier)
                month_vals = {}
                for m in months:
                    col = f"{m} (kWh)"
                    month_vals[m] = r.get(col, None) if col in r.index else None
                theft_val = theft_map.get(str(r.get("ACCOUNT_NUMBER", "")), None)
                # find risk tier if theft_val present
                risk = ""
                if theft_val is not None:
                    if theft_val <= 0.4:
                        risk = "Low"
                    elif theft_val <= 0.7:
                        risk = "Medium"
                    else:
                        risk = "High"
                results.append({
                    "Account No": acc,
                    "Found": True,
                    "Customer Name": r.get("CUSTOMER_NAME", ""),
                    "Billing_Type": r.get("Billing_Type", ""),
                    "Feeder": r.get("Feeder", ""),
                    "DT": r.get("NAME_OF_DT", ""),
                    "METER_NUMBER": r.get("METER_NUMBER", ""),
                    **month_vals,
                    "Theft Probability (avg)": theft_val,
                    "Risk Tier": risk
                })
    report_df = pd.DataFrame(results)
    # order columns: Account No, Found, Customer Name, Billing_Type, Feeder, DT, METER_NUMBER, months..., Theft Probability (avg), Risk Tier
    cols = ["Account No", "Found", "Customer Name", "Billing_Type", "Feeder", "DT", "METER_NUMBER"] + months + ["Theft Probability (avg)", "Risk Tier"]
    # ensure all columns exist
    for c in cols:
        if c not in report_df.columns:
            report_df[c] = None
    report_df = report_df[cols]
    return report_df

st.subheader("Escalations Report")
if st.button("Generate Escalations Report"):
    try:
        report_df = generate_escalations_report(ppm_df, ppd_df, escalations_df, customer_monthly_all, theft_by_account.reset_index(drop=True))
        st.success(f"Escalations report generated: {len(report_df)} records")
        styled_report = report_df.fillna("")
if "Theft Probability (avg)" in styled_report.columns:
    styled_report["Theft Probability (avg)"] = pd.to_numeric(styled_report["Theft Probability (avg)"], errors="coerce")
    styled_report = styled_report.style.format({"Theft Probability (avg)": "{:.3f}"})
st.dataframe(styled_report, use_container_width=True)

        # Excel download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            report_df.to_excel(writer, index=False, sheet_name="Escalations Report")
        buffer.seek(0)
        st.download_button(label="📥 Download Escalations Report (Excel)", data=buffer.getvalue(), file_name="Escalations_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Failed to generate escalations report: {e}")

# Footer
st.markdown("Built by Elvis Ebenuwah for Ikeja Electric. 2025.")

