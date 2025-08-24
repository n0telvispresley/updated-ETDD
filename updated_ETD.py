import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="IE Energy Theft Detection Dashboard", layout="wide")

# Custom converter to preserve exact string values
def preserve_exact_string(value):
    if pd.isna(value) or value is None:
        return ""
    return str(value)

# Function to extract short feeder and DT names
def get_short_name(name, is_dt=False):
    if isinstance(name, str) and name and "-" in name:
        parts = name.split("-")
        if is_dt:
            return parts[-1].strip()  # DT name is last part
        return parts[-1].strip()  # Feeder_Short is last part
    return name if isinstance(name, str) else ""

# File uploader
st.subheader("Upload Excel File")
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.warning("Please upload an Excel file to proceed.")
    st.stop()

# Load Excel file
try:
    sheets = pd.read_excel(
        uploaded_file,
        sheet_name=None,
        converters={
            "Feeder Data": {"Feeder": preserve_exact_string},
            "Transformer Data": {"New Unique DT Nomenclature": preserve_exact_string},
            "Customer Data_PPM": {"NAME_OF_DT": preserve_exact_string, "NAME_OF_FEEDER": preserve_exact_string, "ACCOUNT_NUMBER": preserve_exact_string, "METER_NUMBER": preserve_exact_string, "BUSINESS_UNIT": preserve_exact_string, "UNDERTAKING": preserve_exact_string, "TARIFF": preserve_exact_string, "CUSTOMER_CATEGORY": preserve_exact_string, "METER_STATUS": preserve_exact_string, "ACCOUNT_TYPE": preserve_exact_string, "CUSTOMER_ACCOUNT_TYPE": preserve_exact_string},
            "Customer Data_PPD": {"NAME_OF_DT": preserve_exact_string, "NAME_OF_FEEDER": preserve_exact_string, "ACCOUNT_NUMBER": preserve_exact_string, "METER_NUMBER": preserve_exact_string, "BUSINESS_UNIT": preserve_exact_string, "UNDERTAKING": preserve_exact_string, "TARIFF": preserve_exact_string, "CUSTOMER_CATEGORY": preserve_exact_string, "METER_STATUS": preserve_exact_string, "ACCOUNT_TYPE": preserve_exact_string, "CUSTOMER_ACCOUNT_TYPE": preserve_exact_string},
            "Feeder Band": {"BAND": preserve_exact_string, "Feeder": preserve_exact_string, "Short Name": preserve_exact_string},
            "Customer Tariffs": {"Tariff": preserve_exact_string}
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

# Check if sheets loaded correctly
if any(df is None for df in [feeder_df, dt_df, ppm_df, ppd_df, band_df, tariff_df]):
    st.error("One or more sheets missing.")
    st.stop()

# Validate column names
required_customer_cols = ["NAME_OF_DT", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", "NAME_OF_FEEDER", "BUSINESS_UNIT", "UNDERTAKING", "TARIFF"]
required_dt_cols = ["New Unique DT Nomenclature"]
required_feeder_cols = ["Feeder"]
required_band_cols = ["Feeder", "BAND"]
required_tariff_cols = ["Tariff"]
for df, name, cols in [
    (ppm_df, "Customer Data_PPM", required_customer_cols),
    (ppd_df, "Customer Data_PPD", required_customer_cols),
    (dt_df, "Transformer Data", required_dt_cols),
    (feeder_df, "Feeder Data", required_feeder_cols),
    (band_df, "Feeder Band", required_band_cols),
    (tariff_df, "Customer Tariffs", required_tariff_cols)
]:
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in {name}: {missing_cols}")
        st.stop()

# Validate month columns
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
for df, name in [(ppm_df, "Customer Data_PPM"), (ppd_df, "Customer Data_PPD"), (feeder_df, "Feeder Data"), (dt_df, "Transformer Data")]:
    missing_months = [m for m in months if m not in df.columns]
    if missing_months:
        st.warning(f"Missing month columns in {name}: {missing_months}. Filling with 0.")
        for m in missing_months:
            df[m] = 0

# Handle missing columns
for df, col in [(ppm_df, "TARIFF"), (ppd_df, "TARIFF"), (band_df, "BAND"), (tariff_df, "Tariff")]:
    if col not in df.columns:
        df[col] = ""
if "Short Name" not in band_df.columns:
    band_df["Short Name"] = band_df["Feeder"].apply(lambda x: get_short_name(x))

# Handle Rate column
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

# Normalize names
for col, df in [
    ("Feeder", feeder_df), ("NAME_OF_FEEDER", ppm_df), ("NAME_OF_FEEDER", ppd_df), ("Feeder", band_df),
    ("NAME_OF_DT", ppm_df), ("NAME_OF_DT", ppd_df), ("New Unique DT Nomenclature", dt_df),
    ("TARIFF", ppm_df), ("TARIFF", ppd_df), ("Tariff", tariff_df),
    ("BUSINESS_UNIT", ppm_df), ("BUSINESS_UNIT", ppd_df),
    ("UNDERTAKING", ppm_df), ("UNDERTAKING", ppd_df)
]:
    df[col] = df[col].astype(str).str.strip().str.upper()

# Combine PPM and PPD
ppm_df["Billing_Type"] = "PPM"
ppd_df["Billing_Type"] = "PPD"
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True)
if customer_df.empty:
    st.error("customer_df is empty.")
    st.stop()

# Filter for valid DTs from Transformer Data
valid_dts = set(dt_df["New Unique DT Nomenclature"].astype(str).str.strip().str.upper())
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

# Filter customer_df for valid DTs
customer_df = customer_df[customer_df["NAME_OF_DT"].isin(valid_dts)]
if customer_df.empty:
    st.error("No valid customers after filtering for Transformer Data DTs.")
    st.write("customer_df size:", len(customer_df))
    st.stop()

# Create short names and feeder links
feeder_df["Feeder_Short"] = feeder_df["Feeder"].apply(lambda x: get_short_name(x))
dt_df["DT_Short_Name"] = dt_df["New Unique DT Nomenclature"].apply(lambda x: get_short_name(x, is_dt=True))
dt_df["Feeder"] = dt_df["New Unique DT Nomenclature"].apply(
    lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x and len(x.split("-")) >= 3 else x
)
dt_df["NAME_OF_DT"] = dt_df["New Unique DT Nomenclature"]
customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))
customer_df["Feeder"] = customer_df["NAME_OF_DT"].apply(
    lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x and len(x.split("-")) >= 3 else x
)

# Filter for valid feeders from Feeder Data
valid_feeders = set(feeder_df["Feeder"].astype(str).str.strip().str.upper())
dt_df = dt_df[dt_df["Feeder"].isin(valid_feeders)]
customer_df = customer_df[customer_df["Feeder"].isin(valid_feeders)]
if dt_df.empty or customer_df.empty:
    st.error("No valid data after filtering for Feeder Data feeders.")
    st.write("dt_df size:", len(dt_df))
    st.write("customer_df size:", len(customer_df))
    st.stop()

# Map DTs to feeder tariff rates
dt_df = dt_df.merge(feeder_df[["Feeder", "Tariff_Rate"]], on="Feeder", how="left")
dt_df["Tariff_Rate"] = dt_df["Tariff_Rate"].fillna(209.5)

# Add scores to customer_df
customer_df["meter_status_score"] = np.where(customer_df["METER_STATUS"] == "NOT METERED", 0.9, 0.2)
customer_df["account_type_score"] = np.where(customer_df["ACCOUNT_TYPE"] == "POSTPAID", 0.8, 0.3)
customer_df["customer_account_type_score"] = np.where(customer_df["CUSTOMER_ACCOUNT_TYPE"] == "MD", 0.8, 0.3)
customer_df["billing_type_score"] = np.where(customer_df["Billing_Type"] == "PPD", 0.5, 0.2)
customer_df["customer_category_score"] = customer_df["CUSTOMER_CATEGORY"].map({"RESIDENTIAL": 0.2, "COMMERCIAL": 0.5, "SPECIAL": 0.8}).fillna(0.2)

# Merge tariffs
tariff_matches = customer_df["TARIFF"].isin(tariff_df["Tariff"])
if not tariff_matches.all():
    st.warning(f"Some TARIFF values in customer data not found in Customer Tariffs: {customer_df[~tariff_matches]['TARIFF'].unique()}")
customer_df = customer_df.merge(tariff_df[["Tariff", "Rate (NGN)"]], left_on="TARIFF", right_on="Tariff", how="left")
customer_df["Rate (NGN)"] = customer_df["Rate (NGN)"].fillna(209.5)
customer_df = customer_df.drop(columns=["Tariff"], errors="ignore")

# Streamlit UI: Filters
st.title("Ikeja Electric Energy Theft Detection Dashboard")
st.subheader("Filters")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    bu_options = sorted(customer_df["BUSINESS_UNIT"].unique())
    selected_bu = st.selectbox("Select Business Unit", bu_options)
with col2:
    customer_df_bu = customer_df[customer_df["BUSINESS_UNIT"] == selected_bu]
    ut_options = sorted(customer_df_bu["UNDERTAKING"].unique())
    selected_ut = st.selectbox("Select Undertaking", ut_options)
with col3:
    customer_df_ut = customer_df_bu[customer_df_bu["UNDERTAKING"] == selected_ut]
    feeder_options = sorted(feeder_df["Feeder_Short"].unique())
    if not feeder_options:
        st.error("No feeders available in Feeder Data.")
        st.stop()
    selected_feeder_short = st.selectbox("Select Feeder", feeder_options)
with col4:
    selected_feeder = feeder_df[feeder_df["Feeder_Short"] == selected_feeder_short]["Feeder"].iloc[0]
    dt_df_filtered = dt_df[dt_df["Feeder"] == selected_feeder]
    dt_options = sorted(dt_df_filtered["DT_Short_Name"].unique())
    if not dt_options:
        st.error(f"No DTs available for feeder {selected_feeder_short}.")
        st.write("dt_df Feeder:", sorted(dt_df["Feeder"].unique()))
        st.write("dt_df DT_Short_Name:", sorted(dt_df["DT_Short_Name"].unique()))
        st.write("dt_df Head:", dt_df.head())
        st.stop()
    selected_dt_short = st.selectbox("Select DT", dt_options)
with col5:
    month_options = ["All"] + ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
    selected_month = st.selectbox("Select Month", month_options)

# Dynamic Weight Sliders
st.subheader("Adjust Theft Probability Weights")
colw1, colw2, colw3, colw4 = st.columns(4)
with colw1:
    w_feeder = st.slider("Feeder Score Weight", 0.0, 1.0, 0.15, 0.01)
    w_dt = st.slider("DT Score Weight", 0.0, 1.0, 0.25, 0.01)
with colw2:
    w_meter = st.slider("Meter Status Weight", 0.0, 1.0, 0.15, 0.01)
    w_account = st.slider("Account Type Weight", 0.0, 1.0, 0.15, 0.01)
with colw3:
    w_customer_account = st.slider("Customer Account Type Weight", 0.0, 1.0, 0.15, 0.01)
    w_billing = st.slider("Billing Type Weight", 0.0, 1.0, 0.15, 0.01)
with colw4:
    w_category = st.slider("Customer Category Weight", 0.0, 1.0, 0.10, 0.01)
    w_energy = st.slider("Energy Billed Score Weight", 0.0, 1.0, 0.15, 0.01)

# Normalize weights
total_weight = w_feeder + w_dt + w_meter + w_account + w_customer_account + w_billing + w_category + w_energy
if total_weight == 0:
    st.error("Total weight cannot be zero. Please adjust weights.")
    st.stop()
w_feeder /= total_weight
w_dt /= total_weight
w_meter /= total_weight
w_account /= total_weight
w_customer_account /= total_weight
w_billing /= total_weight
w_category /= total_weight
w_energy /= total_weight

# Filter data by BU and UT
customer_df = customer_df_ut
dt_df = dt_df[dt_df["Feeder"].isin(valid_feeders)]
if dt_df.empty or customer_df.empty:
    st.error("No valid data after BU/UT filtering.")
    st.write("dt_df size:", len(dt_df))
    st.write("customer_df size:", len(customer_df))
    st.stop()

# Debug
if st.checkbox("Debug: Data"):
    st.write("Valid Feeders (Feeder Data):", sorted(feeder_df["Feeder"].unique()))
    st.write("Valid DTs (Transformer Data):", sorted(dt_df["New Unique DT Nomenclature"].unique()))
    st.write("dt_df Feeder:", sorted(dt_df["Feeder"].unique()))
    st.write("dt_df DT_Short_Name:", sorted(dt_df["DT_Short_Name"].unique()))
    st.write("customer_df Columns:", customer_df.columns.tolist())
    st.write("customer_monthly Columns:", customer_monthly.columns.tolist() if 'customer_monthly' in locals() else "customer_monthly not created yet")
    st.write("dt_agg Columns:", dt_agg.columns.tolist() if 'dt_agg' in locals() else "dt_agg not created yet")
    st.write("dt_merged:", dt_merged.head() if 'dt_merged' in locals() else "dt_merged not created yet")
    st.write("feeder_monthly:", feeder_monthly.head() if 'feeder_monthly' in locals() else "feeder_monthly not created yet")
    st.write("feeder_agg:", feeder_agg.head() if 'feeder_agg' in locals() else "feeder_agg not created yet")
    st.write("feeder_merged:", feeder_merged.head() if 'feeder_merged' in locals() else "feeder_merged not created yet")
    st.write("Band Rates:", band_rates)
    st.write("Filtered Customer Count:", len(customer_df))
    st.write("Filtered DT Count:", len(dt_df))
    st.write("Filtered Feeder Count:", len(feeder_df))

# Error Report Download
if not error_report_df.empty:
    st.subheader("Error Report: Invalid Customer DTs")
    st.write("The following customer rows were excluded because their NAME_OF_DT is not in Transformer Data:")
    st.dataframe(error_report_df)
    csv = error_report_df.to_csv(index=False)
    st.download_button(
        label="Download Error Report",
        data=csv,
        file_name="invalid_dts_report.csv",
        mime="text/csv"
    )

# Data preprocessing
for month in months:
    for df, unit in [(feeder_df, 1000), (ppm_df, 1), (ppd_df, 0.001)]:
        col = f"{month} (kWh)"
        if month in df.columns:
            df[col] = pd.to_numeric(df[month], errors="coerce").fillna(0) * unit
        else:
            df[col] = 0
    if month in dt_df.columns:
        dt_df[f"{month} (kWh)"] = pd.to_numeric(df[month], errors="coerce").fillna(0) / 1000
    else:
        dt_df[f"{month} (kWh)"] = 0
for df in [feeder_df, dt_df, ppm_df, ppd_df]:
    df.drop(columns=months, errors="ignore", inplace=True)

# Ensure required columns for melt
required_id_vars = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", "Billing_Type", "Feeder", "Rate (NGN)", "meter_status_score", "account_type_score", "customer_account_type_score", "billing_type_score", "customer_category_score"]
value_vars = [f"{m} (kWh)" for m in months]
missing_id_vars = [col for col in required_id_vars if col not in customer_df.columns]
if missing_id_vars:
    st.error(f"Missing id_vars in customer_df: {missing_id_vars}")
    st.stop()
for col in value_vars:
    if col not in customer_df.columns:
        customer_df[col] = 0

# Melt customer data
try:
    customer_monthly = customer_df.melt(id_vars=required_id_vars, value_vars=value_vars, var_name="month", value_name="billed_kwh")
    customer_monthly["month"] = customer_monthly["month"].str.replace(" (kWh)", "")
except Exception as e:
    st.error(f"Melt failed: {e}")
    st.write("customer_df Columns:", customer_df.columns.tolist())
    st.stop()

# DT consumption
try:
    dt_agg = dt_df.melt(id_vars=["New Unique DT Nomenclature", "DT_Short_Name", "Feeder", "Tariff_Rate"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="total_dt_kwh")
    dt_agg["month"] = dt_agg["month"].str.replace(" (kWh)", "")
except Exception as e:
    st.error(f"DT melt failed: {e}")
    st.write("dt_df Columns:", dt_df.columns.tolist())
    st.stop()

# Apply month filter
if selected_month != "All":
    dt_agg = dt_agg[dt_agg["month"] == selected_month]
    customer_monthly = customer_monthly[customer_monthly["month"] == selected_month]
if dt_agg.empty or customer_monthly.empty:
    st.error(f"No data for selected month {selected_month}.")
    st.write("dt_agg size:", len(dt_agg))
    st.write("customer_monthly size:", len(customer_monthly))
    st.stop()

# Billed energy
try:
    customer_agg = customer_monthly.groupby(["NAME_OF_DT", "month"])["billed_kwh"].sum().reset_index()
    customer_agg.rename(columns={"billed_kwh": "total_billed_kwh"}, inplace=True)
except Exception as e:
    st.error(f"Customer aggregation failed: {e}")
    st.write("customer_monthly:", customer_monthly.head())
    st.stop()

# DT scores
try:
    dt_merged = dt_agg.merge(customer_agg, left_on=["New Unique DT Nomenclature", "month"], right_on=["NAME_OF_DT", "month"], how="left")
    dt_merged["total_billed_kwh"] = dt_merged["total_billed_kwh"].fillna(0)
    dt_merged["dt_score"] = (1 - dt_merged["total_billed_kwh"] / dt_merged["total_dt_kwh"].replace(0, 1)).clip(0, 1)
    dt_merged["energy_lost_kwh"] = dt_merged["total_dt_kwh"] - dt_merged["total_billed_kwh"]
    dt_merged["financial_loss_naira"] = dt_merged["energy_lost_kwh"] * dt_merged["Tariff_Rate"]
except Exception as e:
    st.error(f"DT merge failed: {e}")
    st.write("dt_agg:", dt_agg.head())
    st.write("customer_agg:", customer_agg.head())
    st.stop()

# Feeder scores
try:
    feeder_monthly = feeder_df.melt(id_vars=["Feeder", "Feeder_Short", "Tariff_Rate"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="feeder_energy_kwh")
    feeder_monthly["month"] = feeder_monthly["month"].str.replace(" (kWh)", "")
    if selected_month != "All":
        feeder_monthly = feeder_monthly[feeder_monthly["month"] == selected_month]
    feeder_agg = customer_monthly.groupby(["Feeder", "month"])["billed_kwh"].sum().reset_index().rename(columns={"billed_kwh": "total_billed_kwh"})
    feeder_merged = feeder_monthly.merge(feeder_agg, on=["Feeder", "month"], how="left")
    feeder_merged["total_billed_kwh"] = feeder_merged["total_billed_kwh"].fillna(0)
    feeder_merged["feeder_score"] = (1 - feeder_merged["total_billed_kwh"] / feeder_merged["feeder_energy_kwh"].replace(0, 1)).clip(0, 1)
    feeder_merged["feeder_energy_lost_kwh"] = feeder_merged["feeder_energy_kwh"] - feeder_merged["total_billed_kwh"]
    feeder_merged["feeder_financial_loss_naira"] = feeder_merged["feeder_energy_lost_kwh"] * feeder_merged["Tariff_Rate"]
except Exception as e:
    st.error(f"Feeder merge failed: {e}")
    st.write("feeder_monthly:", feeder_monthly.head())
    st.write("feeder_agg:", feeder_agg.head())
    st.stop()

# Feeder Summary Table
st.subheader("Feeder Summary")
try:
    required_cols = ["Feeder_Short", "month", "feeder_energy_kwh", "total_billed_kwh", "feeder_energy_lost_kwh", "feeder_financial_loss_naira"]
    missing_cols = [col for col in required_cols if col not in feeder_merged.columns]
    if missing_cols:
        st.error(f"Missing columns in feeder_merged: {missing_cols}")
        st.write("feeder_merged Columns:", feeder_merged.columns.tolist())
        st.stop()
    if feeder_merged.empty:
        st.error("feeder_merged is empty.")
        st.write("feeder_monthly:", feeder_monthly.head())
        st.write("feeder_agg:", feeder_agg.head())
        st.stop()
    feeder_summary = feeder_merged[required_cols].groupby(["Feeder_Short", "month"]).sum().reset_index()
    feeder_summary.columns = ["Feeder", "Month", "Energy Supplied (kWh)", "Energy Billed (kWh)", "Energy Unaccounted For (kWh)", "Financial Loss (NGN)"]
    if not feeder_summary.empty:
        st.dataframe(feeder_summary.style.format({
            "Energy Supplied (kWh)": "{:.2f}",
            "Energy Billed (kWh)": "{:.2f}",
            "Energy Unaccounted For (kWh)": "{:.2f}",
            "Financial Loss (NGN)": "{:.2f}"
        }))
    else:
        st.warning("No feeder summary data available.")
        st.write("feeder_merged:", feeder_merged.head())
except Exception as e:
    st.error(f"Feeder summary failed: {e}")
    st.write("feeder_merged:", feeder_merged.head() if 'feeder_merged' in locals() else "feeder_merged not created")
    st.stop()

# DT Summary Table
st.subheader(f"DT Summary for {selected_feeder_short}")
try:
    dt_summary = dt_merged[dt_merged["Feeder"] == selected_feeder].groupby(["DT_Short_Name", "month"]).agg({
        "total_dt_kwh": "sum",
        "total_billed_kwh": "sum",
        "energy_lost_kwh": "sum",
        "financial_loss_naira": "sum"
    }).reset_index()
    dt_summary.columns = ["DT", "Month", "Energy Supplied (kWh)", "Energy Billed (kWh)", "Energy Unaccounted For (kWh)", "Financial Loss (NGN)"]
    if not dt_summary.empty:
        st.dataframe(dt_summary.style.format({
            "Energy Supplied (kWh)": "{:.2f}",
            "Energy Billed (kWh)": "{:.2f}",
            "Energy Unaccounted For (kWh)": "{:.2f}",
            "Financial Loss (NGN)": "{:.2f}"
        }))
    else:
        st.warning(f"No DT summary data for {selected_feeder_short}.")
except Exception as e:
    st.error(f"DT summary failed: {e}")
    st.write("dt_merged:", dt_merged.head())
    st.stop()

# DT Theft Probability Heatmap
st.subheader("DT Theft Probability Heatmap")
try:
    filtered_dt_agg = dt_agg[dt_agg["Feeder"] == selected_feeder]
    if filtered_dt_agg.empty:
        st.error(f"No DT data for feeder {selected_feeder_short}.")
        st.write("dt_agg:", dt_agg.head())
        st.write("dt_merged:", dt_merged.head())
        st.stop()
    filtered_dt_agg = filtered_dt_agg.merge(dt_merged[["New Unique DT Nomenclature", "month", "dt_score"]], on=["New Unique DT Nomenclature", "month"], how="left")
    filtered_dt_agg["dt_score"] = filtered_dt_agg["dt_score"].fillna(0)
    dt_pivot = filtered_dt_agg.pivot_table(index="DT_Short_Name", columns="month", values="dt_score", aggfunc="mean")
    if not dt_pivot.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(dt_pivot, cmap="YlOrRd", cbar_kws={"label": "DT Theft Score"})
        plt.title(f"DT Theft Score for {selected_feeder_short} ({selected_month})")
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.error(f"No DT data for {selected_feeder_short} after pivoting.")
        st.write("filtered_dt_agg:", filtered_dt_agg.head())
        st.stop()
except Exception as e:
    st.error(f"DT heatmap failed: {e}")
    st.write("filtered_dt_agg:", filtered_dt_agg.head() if 'filtered_dt_agg' in locals() else "Not created")
    st.stop()

# Customer scores
try:
    customer_monthly["energy_billed_score"] = (1 - customer_monthly["billed_kwh"] / customer_monthly["billed_kwh"].replace(0, 1).max()).clip(0, 1)
    customer_monthly = customer_monthly.merge(feeder_merged[["Feeder", "month", "feeder_score"]], on=["Feeder", "month"], how="left")
    customer_monthly = customer_monthly.merge(dt_merged[["New Unique DT Nomenclature", "month", "dt_score"]], left_on=["NAME_OF_DT", "month"], right_on=["New Unique DT Nomenclature", "month"], how="left")
    customer_monthly["feeder_score"] = customer_monthly["feeder_score"].fillna(0)
    customer_monthly["dt_score"] = customer_monthly["dt_score"].fillna(0)
    customer_monthly["theft_probability"] = (
        w_feeder * customer_monthly["feeder_score"] +
        w_dt * customer_monthly["dt_score"] +
        w_meter * customer_monthly["meter_status_score"] +
        w_account * customer_monthly["account_type_score"] +
        w_customer_account * customer_monthly["customer_account_type_score"] +
        w_billing * customer_monthly["billing_type_score"] +
        w_category * customer_monthly["customer_category_score"] +
        w_energy * customer_monthly["energy_billed_score"]
    ).clip(0, 1)
    customer_monthly["risk_tier"] = pd.cut(customer_monthly["theft_probability"], bins=[0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"], include_lowest=True)
except Exception as e:
    st.error(f"Customer scoring failed: {e}")
    st.write("customer_monthly:", customer_monthly.head())
    st.stop()

# Customer Heatmap
st.subheader("Theft Analysis")
try:
    filtered_customers = customer_monthly[customer_monthly["DT_Short_Name"] == selected_dt_short]
    num_customers = st.number_input("Number of high-risk customers (0 for all)", min_value=0, value=10, step=1)
    if num_customers > 0:
        filtered_customers = filtered_customers.sort_values(by="theft_probability", ascending=False).head(num_customers)
    pivot_data = filtered_customers.pivot_table(index="ACCOUNT_NUMBER", columns="month", values="theft_probability", aggfunc="mean")
    if not pivot_data.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, cmap="YlOrRd", vmin=0, vmax=1, cbar_kws={"label": "Theft Probability"})
        plt.title(f"Theft Probability for {selected_dt_short} ({selected_feeder_short}, {selected_month})")
        st.pyplot(plt.gcf())
        plt.close()
    else:
        st.warning(f"No customer data for {selected_dt_short} ({selected_month}).")
        st.write("filtered_customers:", filtered_customers.head())
except Exception as e:
    st.error(f"Customer heatmap failed: {e}")
    st.write("filtered_customers:", filtered_customers.head() if 'filtered_customers' in locals() else "Not created")
    st.stop()

# Customer List
st.subheader(f"Customers under {selected_dt_short} ({selected_feeder_short}, {selected_month})")
try:
    if selected_month == "All":
        month_customers = filtered_customers.groupby(["ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", "Billing_Type"]).agg({
            "billed_kwh": "sum",
            "theft_probability": "mean",
            "risk_tier": lambda x: pd.Series(x).mode()[0]
        }).reset_index()
    else:
        month_customers = filtered_customers[filtered_customers["month"] == selected_month]
    if not month_customers.empty:
        display_columns = ["ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "billed_kwh", "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", "Billing_Type", "theft_probability", "risk_tier"]
        missing_cols = [col for col in display_columns if col not in month_customers.columns]
        if missing_cols:
            st.error(f"Missing columns in month_customers: {missing_cols}")
            st.write("month_customers Columns:", month_customers.columns.tolist())
            st.stop()
        styled_df = month_customers[display_columns].style.format({
            "billed_kwh": "{:.2f}",
            "theft_probability": "{:.3f}"
        }).highlight_max(subset=["theft_probability"], color="lightcoral")
        st.dataframe(styled_df)
    else:
        st.warning(f"No customers for {selected_dt_short} ({selected_month}).")
        st.write("filtered_customers:", filtered_customers.head())
except Exception as e:
    st.error(f"Customer list failed: {e}")
    st.write("filtered_customers:", filtered_customers.head())
    st.stop()

# CSV Export
st.subheader("Export Customer Data")
if not month_customers.empty:
    csv = month_customers[display_columns].to_csv(index=False)
    st.download_button(label=f"Download Customer List ({selected_month})", data=csv, file_name=f"theft_analysis_{selected_dt_short}_{selected_feeder_short}_{selected_month}.csv", mime="text/csv")

# Footer
st.markdown("Built by Elvis for Ikeja Electric SIWES III. August 2025.")
