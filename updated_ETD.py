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
        return ""  # Convert NaN/Null/None to empty string
    return str(value)  # Preserve exact string, including apostrophes

# Function to extract DT short name (part after last hyphen)
def get_dt_short_name(dt_name):
    if isinstance(dt_name, str) and dt_name and "-" in dt_name:
        return dt_name.split("-")[-1].strip()
    return dt_name if isinstance(dt_name, str) else ""

# File uploader
st.subheader("Upload Excel File")
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.warning("Please upload an Excel file to proceed.")
    st.stop()

# Load Excel file with converters
try:
    sheets = pd.read_excel(
        uploaded_file,
        sheet_name=None,
        converters={
            "Feeder Data": {"Feeder": preserve_exact_string, "Feeder No": preserve_exact_string},
            "Transformer Data": {"New Unique DT Nomenclature": preserve_exact_string, "DT Number": preserve_exact_string, "Ownership": preserve_exact_string, "Connection Status": preserve_exact_string},
            "Customer Data_PPM": {"NAME_OF_DT": preserve_exact_string, "NAME_OF_FEEDER": preserve_exact_string, "ACCOUNT_NUMBER": preserve_exact_string, "METER_NUMBER": preserve_exact_string, "BUSINESS_UNIT": preserve_exact_string, "UNDERTAKING": preserve_exact_string, "TARIFF": preserve_exact_string, "CUSTOMER_CATEGORY": preserve_exact_string, "METER_STATUS": preserve_exact_string, "ACCOUNT_TYPE": preserve_exact_string, "CUSTOMER_ACCOUNT_TYPE": preserve_exact_string},
            "Customer Data_PPD": {"NAME_OF_DT": preserve_exact_string, "NAME_OF_FEEDER": preserve_exact_string, "ACCOUNT_NUMBER": preserve_exact_string, "METER_NUMBER": preserve_exact_string, "BUSINESS_UNIT": preserve_exact_string, "UNDERTAKING": preserve_exact_string, "TARIFF": preserve_exact_string, "CUSTOMER_CATEGORY": preserve_exact_string, "METER_STATUS": preserve_exact_string, "ACCOUNT_TYPE": preserve_exact_string, "CUSTOMER_ACCOUNT_TYPE": preserve_exact_string},
            "Feeder Band": {"BAND": preserve_exact_string, "Feeder": preserve_exact_string, "Short Name": preserve_exact_string},
            "Customer Tariffs": {"Tariff": preserve_exact_string, "Rate (₦)": preserve_exact_string}
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
if feeder_df is None or dt_df is None or ppm_df is None or ppd_df is None or band_df is None or tariff_df is None:
    st.error("One or more sheets (Feeder Data, Transformer Data, Customer Data_PPM, Customer Data_PPD, Feeder Band, Customer Tariffs) not found.")
    st.stop()

# Validate column names
required_customer_cols = ["NAME_OF_DT", "ACCOUNT_NUMBER", "CUSTOMER_NAME", "ADDRESS", "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", "NAME_OF_FEEDER", "BUSINESS_UNIT", "UNDERTAKING"]
for df, name in [(ppm_df, "Customer Data_PPM"), (ppd_df, "Customer Data_PPD")]:
    missing_cols = [col for col in required_customer_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in {name}: {missing_cols}. Available columns: {df.columns.tolist()}")
        st.stop()
    if "TARIFF" not in df.columns:
        st.warning(f"Column 'TARIFF' not found in {name}. Setting to empty string.")
        df["TARIFF"] = ""
if "BAND" not in band_df.columns:
    st.error(f"Column 'BAND' not found in Feeder Band. Available columns: {band_df.columns.tolist()}")
    band_df["BAND"] = ""  # Fallback: empty BAND column
if "Feeder" not in band_df.columns:
    st.error(f"Column 'Feeder' not found in Feeder Band. Available columns: {band_df.columns.tolist()}")
    st.stop()
if "Short Name" not in band_df.columns:
    st.error(f"Column 'Short Name' not found in Feeder Band. Available columns: {band_df.columns.tolist()}")
    st.stop()
if "Tariff" not in tariff_df.columns:
    st.error(f"Column 'Tariff' not found in Customer Tariffs. Available columns: {tariff_df.columns.tolist()}")
    tariff_df["Tariff"] = ""
if "Rate (₦)" not in tariff_df.columns:
    st.warning(f"Column 'Rate (₦)' not found in Customer Tariffs. Available columns: {tariff_df.columns.tolist()}. Renaming 'Rate' if present or creating with default 209.5.")
    if "Rate" in tariff_df.columns:
        tariff_df["Rate (₦)"] = tariff_df["Rate"]
    else:
        tariff_df["Rate (₦)"] = 209.5

# Normalize feeder, DT, and tariff names
feeder_df["Feeder"] = feeder_df["Feeder"].str.strip().str.upper()
ppm_df["NAME_OF_FEEDER"] = ppm_df["NAME_OF_FEEDER"].str.strip().str.upper()
ppd_df["NAME_OF_FEEDER"] = ppd_df["NAME_OF_FEEDER"].str.strip().str.upper()
band_df["Feeder"] = band_df["Feeder"].str.strip().str.upper()
ppm_df["NAME_OF_DT"] = ppm_df["NAME_OF_DT"].str.strip().str.upper()
ppd_df["NAME_OF_DT"] = ppd_df["NAME_OF_DT"].str.strip().str.upper()
dt_df["New Unique DT Nomenclature"] = dt_df["New Unique DT Nomenclature"].str.strip().str.upper()
ppm_df["TARIFF"] = ppm_df["TARIFF"].str.strip().str.upper()
ppd_df["TARIFF"] = ppd_df["TARIFF"].str.strip().str.upper()
tariff_df["Tariff"] = tariff_df["Tariff"].str.strip().str.upper()

# Combine PPM and PPD into customer_df
ppm_df["Billing_Type"] = "PPM"
ppd_df["Billing_Type"] = "PPD"
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True)

# Check if customer_df is empty
if customer_df.empty:
    st.error("customer_df is empty. Check if Customer Data_PPM and Customer Data_PPD have valid data.")
    st.write("PPM rows:", len(ppm_df))
    st.write("PPD rows:", len(ppd_df))
    st.stop()

# Create DT short name mapping
customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(get_dt_short_name)
dt_df["DT_Short_Name"] = dt_df["New Unique DT Nomenclature"].apply(get_dt_short_name)

# Merge with tariffs
tariff_merge = customer_df.merge(tariff_df[["Tariff", "Rate (₦)"]], left_on="TARIFF", right_on="Tariff", how="left")
customer_df["Rate (₦)"] = tariff_merge["Rate (₦)"].fillna(209.5)
customer_df = customer_df.drop(columns=["Tariff"], errors="ignore")

# Debug: Check tariff merge
if st.checkbox("Debug: Tariff merge"):
    st.write("customer_df TARIFF unique values:", sorted(customer_df["TARIFF"].dropna().astype(str).unique()))
    st.write("tariff_df Tariff unique values:", sorted(tariff_df["Tariff"].dropna().astype(str).unique()))
    st.write("Rows in customer_df with Rate (₦) after merge:", len(customer_df[customer_df["Rate (₦)"].notna()]))
    st.write("Rows in customer_df with null Rate (₦):", len(customer_df[customer_df["Rate (₦)"].isna()]))
    st.write("Sample Rate (₦) values:", customer_df["Rate (₦)"].head().tolist())

# Data preprocessing
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]

# Convert units
# Feeder Data: MWh to kWh
for month in months:
    if month in feeder_df.columns:
        feeder_df[month + " (kWh)"] = pd.to_numeric(feeder_df[month], errors="coerce") * 1000
        feeder_df[month + " (kWh)"] = feeder_df[month + " (kWh)"].fillna(0)
    else:
        st.warning(f"Month {month} not found in Feeder Data. Creating zero-filled column.")
        feeder_df[month + " (kWh)"] = 0
feeder_df = feeder_df.drop(columns=months, errors="ignore")

# Transformer Data: Wh to kWh
for month in months:
    if month in dt_df.columns:
        dt_df[month + " (kWh)"] = pd.to_numeric(dt_df[month], errors="coerce") / 1000
        dt_df[month + " (kWh)"] = dt_df[month + " (kWh)"].fillna(0)
    else:
        st.warning(f"Month {month} not found in Transformer Data. Creating zero-filled column.")
        dt_df[month + " (kWh)"] = 0
dt_df = dt_df.drop(columns=months, errors="ignore")

# Customer Data: kWh for PPM, Wh to kWh for PPD
for month in months:
    if month in ppm_df.columns:
        ppm_df[month + " (kWh)"] = pd.to_numeric(ppm_df[month], errors="coerce").fillna(0)
    else:
        st.warning(f"Month {month} not found in Customer Data_PPM. Creating zero-filled column.")
        ppm_df[month + " (kWh)"] = 0
    if month in ppd_df.columns:
        ppd_df[month + " (kWh)"] = pd.to_numeric(ppd_df[month], errors="coerce") / 1000
        ppd_df[month + " (kWh)"] = ppd_df[month + " (kWh)"].fillna(0)
    else:
        st.warning(f"Month {month} not found in Customer Data_PPD. Creating zero-filled column.")
        ppd_df[month + " (kWh)"] = 0
ppm_df = ppm_df.drop(columns=months, errors="ignore")
ppd_df = ppd_df.drop(columns=months, errors="ignore")
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True)
customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(get_dt_short_name)

# Debug: Validate customer_df columns before melt
if st.checkbox("Debug: Customer Data before melt"):
    st.write("customer_df columns:", customer_df.columns.tolist())
    st.write("customer_df rows:", len(customer_df))
    st.write("Sample NAME_OF_DT values:", customer_df["NAME_OF_DT"].head().tolist())
    st.write("Sample DT_Short_Name values:", customer_df["DT_Short_Name"].head().tolist())
    st.write("Sample NAME_OF_FEEDER values:", customer_df["NAME_OF_FEEDER"].head().tolist())
    st.write("Sample Rate (₦) values:", customer_df["Rate (₦)"].head().tolist())
    st.write("Month columns present:", [col for col in customer_df.columns if col.endswith(" (kWh)")])
    st.write("tariff_df columns:", tariff_df.columns.tolist())
    st.write("Sample tariff_df Tariff values:", tariff_df["Tariff"].head().tolist())
    st.write("Sample tariff_df Rate (₦) values:", tariff_df["Rate (₦)"].head().tolist())

# Validate columns for melt
required_id_vars = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "Rate (₦)", "CUSTOMER_NAME", "ADDRESS", "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", "Billing_Type", "NAME_OF_FEEDER", "BUSINESS_UNIT", "UNDERTAKING"]
value_vars = [m + " (kWh)" for m in months]
missing_id_vars = [col for col in required_id_vars if col not in customer_df.columns]
missing_value_vars = [col for col in value_vars if col not in customer_df.columns]
if missing_id_vars or missing_value_vars:
    st.error(f"Missing columns in customer_df for melt. Missing id_vars: {missing_id_vars}, Missing value_vars: {missing_value_vars}")
    st.stop()

# Verify unbilled energy accuracy
dt_df["Calculated Avg Monthly Unbilled Energy (kWh)"] = sum(dt_df[m + " (kWh)"] for m in months) / 6
dt_df["Provided Avg Monthly Unbilled Energy (kWh)"] = pd.to_numeric(dt_df["Avg Monthly Unbilled Energy"], errors="coerce") / 1000
dt_df["Provided Avg Monthly Unbilled Energy (kWh)"] = dt_df["Provided Avg Monthly Unbilled Energy (kWh)"].fillna(0)
dt_df["Unbilled Energy Discrepancy (kWh)"] = np.abs(dt_df["Calculated Avg Monthly Unbilled Energy (kWh)"] - dt_df["Provided Avg Monthly Unbilled Energy (kWh)"])
dt_df["Unbilled Energy Accuracy"] = dt_df["Unbilled Energy Discrepancy (kWh)"] < 1

# Filter inactive DTs
dt_df["Has Energy"] = dt_df[[m + " (kWh)" for m in months]].gt(0).any(axis=1)
dt_df["Flag"] = (dt_df["Connection Status"] == "Not Connected") & (dt_df["Has Energy"])
dt_df = dt_df[(dt_df["Connection Status"] == "Connected") | dt_df["Flag"]]

# Calculate scores
customer_df["meter_status_score"] = np.where(customer_df["METER_STATUS"] == "Not Metered", 0.9, 0.2)
customer_df["account_type_score"] = np.where(customer_df["ACCOUNT_TYPE"] == "Postpaid", 0.8, 0.3)
customer_df["customer_account_type_score"] = np.where(customer_df["CUSTOMER_ACCOUNT_TYPE"] == "MD", 0.8, 0.3)
customer_df["billing_type_score"] = np.where(customer_df["Billing_Type"] == "PPD", 0.5, 0.2)
customer_df["customer_category_score"] = customer_df["CUSTOMER_CATEGORY"].map({"Residential": 0.2, "Commercial": 0.5, "Special": 0.8}).fillna(0.2)

# Calculate total DT consumption per DT per month
dt_agg = dt_df.melt(id_vars=["New Unique DT Nomenclature", "DT Number", "Flag", "DT_Short_Name"], value_vars=[m + " (kWh)" for m in months], var_name="month", value_name="total_dt_kwh")
dt_agg["month"] = dt_agg["month"].str.replace(" (kWh)", "")

# Calculate total billed energy per DT per month
customer_monthly = customer_df.melt(id_vars=["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "Rate (₦)", "CUSTOMER_NAME", "ADDRESS", "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", "Billing_Type", "NAME_OF_FEEDER", "BUSINESS_UNIT", "UNDERTAKING"], value_vars=[m + " (kWh)" for m in months], var_name="month", value_name="billed_kwh")
customer_monthly["month"] = customer_monthly["month"].str.replace(" (kWh)", "")
customer_agg = customer_monthly.groupby(["NAME_OF_DT", "month"])["billed_kwh"].sum().reset_index()
customer_agg.rename(columns={"billed_kwh": "total_billed_kwh"}, inplace=True)

# Calculate DT scores
dt_merged = dt_agg.merge(customer_agg, left_on=["New Unique DT Nomenclature", "month"], right_on=["NAME_OF_DT", "month"], how="left")
dt_merged["total_billed_kwh"] = dt_merged["total_billed_kwh"].fillna(0)
dt_merged["dt_score"] = (1 - dt_merged["total_billed_kwh"] / dt_merged["total_dt_kwh"].replace(0, 1)).clip(0, 1)
dt_merged["energy_lost_kwh"] = dt_merged["total_dt_kwh"] - dt_merged["total_billed_kwh"]
dt_merged["financial_loss_naira"] = dt_merged["energy_lost_kwh"] * 209.5

# Calculate feeder scores using customer_df's NAME_OF_FEEDER
feeder_monthly = feeder_df.melt(id_vars=["Feeder"], value_vars=[m + " (kWh)" for m in months], var_name="month", value_name="feeder_energy_kwh")
feeder_monthly["month"] = feeder_monthly["month"].str.replace(" (kWh)", "")
feeder_agg = customer_monthly.groupby(["NAME_OF_FEEDER", "month"])["billed_kwh"].sum().reset_index().rename(columns={"billed_kwh": "total_billed_kwh"})
feeder_merged = feeder_monthly.merge(feeder_agg, left_on=["Feeder", "month"], right_on=["NAME_OF_FEEDER", "month"], how="left")
feeder_merged["total_billed_kwh"] = feeder_merged["total_billed_kwh"].fillna(0)
feeder_merged["feeder_score"] = (1 - feeder_merged["total_billed_kwh"] / feeder_merged["feeder_energy_kwh"].replace(0, 1)).clip(0, 1)
feeder_merged["feeder_energy_lost_kwh"] = feeder_merged["feeder_energy_kwh"] - feeder_merged["total_billed_kwh"]
feeder_merged = feeder_merged.merge(band_df[["Feeder", "Short Name", "BAND"]], on="Feeder", how="left")
feeder_merged["BAND"] = feeder_merged["BAND"].fillna("")  # Fallback for missing BAND
feeder_merged["feeder_financial_loss_naira"] = feeder_merged["feeder_energy_lost_kwh"] * 209.5

# Debug: Check merge inputs
if st.checkbox("Debug: Merge inputs for customer_monthly"):
    st.write("customer_monthly NAME_OF_DT unique values:", sorted(customer_monthly["NAME_OF_DT"].dropna().astype(str).unique()))
    st.write("dt_merged New Unique DT Nomenclature unique values:", sorted(dt_merged["New Unique DT Nomenclature"].dropna().astype(str).unique()))
    st.write("customer_monthly NAME_OF_FEEDER unique values:", sorted(customer_monthly["NAME_OF_FEEDER"].dropna().astype(str).unique()))
    st.write("feeder_merged Feeder unique values:", sorted(feeder_merged["Feeder"].dropna().astype(str).unique()))
    st.write("customer_monthly month unique values:", customer_monthly["month"].unique().tolist())
    st.write("dt_merged month unique values:", dt_merged["month"].unique().tolist())
    st.write("feeder_merged BAND values:", feeder_merged["BAND"].dropna().unique().tolist())

# Debug: Check feeder_summary merge inputs
if st.checkbox("Debug: Feeder summary merge"):
    st.write("feeder_merged[month == 'JUN'] Feeder values:", sorted(feeder_merged[feeder_merged["month"] == "JUN"]["Feeder"].dropna().astype(str).unique()))
    st.write("band_df Feeder values:", sorted(band_df["Feeder"].dropna().astype(str).unique()))

# Handle MD-owned DTs
dt_no_customers = dt_merged[~dt_merged["New Unique DT Nomenclature"].isin(customer_df["NAME_OF_DT"])]
md_customers = dt_no_customers[["New Unique DT Nomenclature", "DT Number", "month", "dt_score", "Flag", "DT_Short_Name"]].copy()
md_customers["ACCOUNT_NUMBER"] = md_customers["DT Number"]
md_customers["METER_NUMBER"] = md_customers["DT Number"]
md_customers["CUSTOMER_NAME"] = md_customers["New Unique DT Nomenclature"]
md_customers["ADDRESS"] = "MD-Owned DT"
md_customers["billed_kwh"] = 0
md_customers["METER_STATUS"] = "Not Metered"
md_customers["ACCOUNT_TYPE"] = "Postpaid"
md_customers["CUSTOMER_ACCOUNT_TYPE"] = "MD"
md_customers["Billing_Type"] = "PPD"
md_customers["CUSTOMER_CATEGORY"] = "Special"
md_customers["Rate (₦)"] = 209.5
md_customers["NAME_OF_FEEDER"] = "Unknown"  # Will be filtered out unless matched
md_customers["BUSINESS_UNIT"] = "MD"
md_customers["UNDERTAKING"] = "MD"
md_customers["theft_probability"] = md_customers["dt_score"]
md_customers["meter_status_score"] = 0.9
md_customers["account_type_score"] = 0.8
md_customers["customer_account_type_score"] = 0.8
md_customers["billing_type_score"] = 0.5
md_customers["customer_category_score"] = 0.8
md_customers["energy_billed_score"] = 0.0

# Append MD-owned DTs
customer_monthly = pd.concat([customer_monthly, md_customers], ignore_index=True)

# Calculate customer scores
customer_monthly = customer_monthly.merge(feeder_merged[["Feeder", "month", "feeder_score"]], left_on=["NAME_OF_FEEDER", "month"], right_on=["Feeder", "month"], how="left")
customer_monthly = customer_monthly.merge(dt_merged[["New Unique DT Nomenclature", "month", "dt_score"]], left_on=["NAME_OF_DT", "month"], right_on=["New Unique DT Nomenclature", "month"], how="left")
customer_monthly["feeder_score"] = customer_monthly["feeder_score"].fillna(0)
customer_monthly["dt_score"] = customer_monthly["dt_score"].fillna(0)
customer_monthly["energy_billed_score"] = (1 - customer_monthly["billed_kwh"] / customer_monthly["billed_kwh"].replace(0, 1).max()).clip(0, 1)
customer_monthly["theft_probability"] = (
    0.15 * customer_monthly["feeder_score"] +
    0.25 * customer_monthly["dt_score"] +
    0.15 * customer_monthly["meter_status_score"] +
    0.15 * customer_monthly["account_type_score"] +
    0.15 * customer_monthly["customer_account_type_score"] +
    0.15 * customer_monthly["billing_type_score"] +
    0.10 * customer_monthly["customer_category_score"] +
    0.15 * customer_monthly["energy_billed_score"]
).clip(0, 1)

# Add risk tiers
customer_monthly["risk_tier"] = pd.cut(
    customer_monthly["theft_probability"],
    bins=[0, 0.4, 0.7, 1.0],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

# Streamlit UI
st.title("Ikeja Electric Energy Theft Detection Dashboard")
st.markdown("Detect high-risk buildings and MD-owned DTs using multi-month data (January–June 2025).")

# Filters
st.subheader("Filters")
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    business_unit_options = ["All"] + sorted(customer_df["BUSINESS_UNIT"].dropna().astype(str).unique())
    selected_business_unit = st.selectbox("Select Business Unit", business_unit_options)
with col2:
    undertaking_options = ["All"] + sorted(customer_df["UNDERTAKING"].dropna().astype(str).unique())
    selected_undertaking = st.selectbox("Select Undertaking", undertaking_options)
with col3:
    band_options = ["All"] + sorted(band_df["BAND"].dropna().astype(str).unique())
    selected_band = st.selectbox("Select Band", band_options)

# Apply Business Unit and Undertaking filters first
filtered_customer_df = customer_df.copy()
if selected_business_unit != "All":
    filtered_customer_df = filtered_customer_df[filtered_customer_df["BUSINESS_UNIT"] == selected_business_unit]
if selected_undertaking != "All":
    filtered_customer_df = filtered_customer_df[filtered_customer_df["UNDERTAKING"] == selected_undertaking]
if filtered_customer_df.empty:
    st.error("No data available after applying Business Unit and Undertaking filters.")
    st.write("Selected Business Unit:", selected_business_unit)
    st.write("Selected Undertaking:", selected_undertaking)
    st.write("Available BUSINESS_UNIT values:", sorted(customer_df["BUSINESS_UNIT"].dropna().astype(str).unique()))
    st.write("Available UNDERTAKING values:", sorted(customer_df["UNDERTAKING"].dropna().astype(str).unique()))
    st.stop()
if "NAME_OF_FEEDER" not in filtered_customer_df.columns:
    st.error("NAME_OF_FEEDER column missing in filtered_customer_df.")
    st.write("filtered_customer_df columns:", filtered_customer_df.columns.tolist())
    st.stop()

# Debug: Check filtered_customer_df
if st.checkbox("Debug: Filtered customer data"):
    st.write("filtered_customer_df rows:", len(filtered_customer_df))
    st.write("filtered_customer_df columns:", filtered_customer_df.columns.tolist())
    st.write("Sample NAME_OF_FEEDER values:", filtered_customer_df["NAME_OF_FEEDER"].head().tolist())
    st.write("Unique NAME_OF_FEEDER values:", sorted(filtered_customer_df["NAME_OF_FEEDER"].dropna().astype(str).unique()))
    st.write("Unique DT_Short_Name values:", sorted(filtered_customer_df["DT_Short_Name"].dropna().astype(str).unique()))

filtered_dt_df = dt_df[dt_df["New Unique DT Nomenclature"].isin(filtered_customer_df["NAME_OF_DT"]) | dt_df["Flag"]]

with col4:
    if selected_band == "All":
        feeder_options = band_df[band_df["Feeder"].isin(filtered_customer_df["NAME_OF_FEEDER"])]["Short Name"].dropna().astype(str).tolist()
        if not feeder_options:  # Fallback if no matches
            feeder_options = band_df["Short Name"].dropna().astype(str).tolist()
            st.warning("No feeders match NAME_OF_FEEDER. Showing all feeders from Feeder Band.")
    else:
        feeder_options = band_df[(band_df["BAND"] == selected_band) & (band_df["Feeder"].isin(filtered_customer_df["NAME_OF_FEEDER"]))]["Short Name"].dropna().astype(str).tolist()
        if not feeder_options:  # Fallback if no matches
            feeder_options = band_df[band_df["BAND"] == selected_band]["Short Name"].dropna().astype(str).tolist()
            st.warning(f"No feeders match NAME_OF_FEEDER for band {selected_band}. Showing all feeders for this band.")
    if not feeder_options:
        st.error("No feeders available for the selected band, business unit, or undertaking.")
        st.stop()
    feeder_options = sorted(feeder_options)
    selected_feeder_short = st.selectbox("Select Feeder", feeder_options)
    selected_feeder = band_df[band_df["Short Name"] == selected_feeder_short]["Feeder"].iloc[0] if selected_feeder_short else None
with col5:
    dt_options = filtered_customer_df[filtered_customer_df["NAME_OF_FEEDER"] == selected_feeder]["DT_Short_Name"].dropna().astype(str).unique().tolist()
    dt_options += [dt for dt in filtered_dt_df[filtered_dt_df["Flag"]]["DT_Short_Name"].dropna().astype(str).tolist() if dt not in dt_options]
    dt_options = sorted(dt_options)
    if not dt_options:
        st.error(f"No DTs available for feeder {selected_feeder_short}. Check NAME_OF_FEEDER in Customer Data.")
        st.stop()
    # Map short names to full names for internal use
    dt_short_to_full = {get_dt_short_name(dt): dt for dt in filtered_customer_df[filtered_customer_df["NAME_OF_FEEDER"] == selected_feeder]["NAME_OF_DT"].dropna().astype(str).unique()}
    dt_short_to_full.update({get_dt_short_name(dt): dt for dt in filtered_dt_df[filtered_dt_df["Flag"]]["New Unique DT Nomenclature"].dropna().astype(str)})
    dt_options_display = [f"{dt} (FLAG: Inactive with Energy)" if dt_short_to_full.get(dt) in filtered_dt_df[filtered_dt_df["Flag"]]["New Unique DT Nomenclature"].tolist() else dt for dt in dt_options]
    selected_dt_short = st.selectbox("Select DT", dt_options_display)
    selected_dt_name = dt_short_to_full.get(str(selected_dt_short).replace(" (FLAG: Inactive with Energy)", ""), "") if selected_dt_short else ""
with col6:
    selected_month = st.selectbox("Select Month", months)

# Feeder-Level Summary
st.subheader("Feeder-Level Loss Summary")
feeder_summary = feeder_merged[feeder_merged["month"] == selected_month].merge(band_df[["Feeder", "Short Name", "BAND"]], on="Feeder", how="left")
if "BAND" not in feeder_summary.columns:
    st.warning("BAND column missing in feeder_summary after merge. Adding empty BAND column.")
    feeder_summary["BAND"] = ""
feeder_summary["BAND"] = feeder_summary["BAND"].fillna("")  # Fallback for missing BAND
feeder_summary = feeder_summary[feeder_summary["Feeder"].isin(filtered_customer_df["NAME_OF_FEEDER"])]
if feeder_summary.empty:
    st.warning("No feeders match the selected filters. Check NAME_OF_FEEDER in Customer Data and Feeder in Feeder Band.")
else:
    if "BAND" in feeder_summary.columns and feeder_summary["BAND"].notna().any():
        feeder_summary = feeder_summary.sort_values("BAND")
    st.dataframe(feeder_summary[["Short Name", "feeder_energy_lost_kwh", "feeder_financial_loss_naira"]].style.format({"feeder_energy_lost_kwh": "{:,.2f}", "feeder_financial_loss_naira": "₦{:,.2f}"}))

# DT Heatmap
st.subheader("DT Theft Probability Heatmap")
dt_pivot = dt_merged[dt_merged["New Unique DT Nomenclature"].isin(filtered_customer_df[filtered_customer_df["NAME_OF_FEEDER"] == selected_feeder]["NAME_OF_DT"].unique()) | dt_merged["Flag"]].pivot_table(index="New Unique DT Nomenclature", columns="month", values="dt_score", aggfunc="mean")
if not dt_pivot.empty:
    plt.figure(figsize=(10, 8))
    sns.heatmap(dt_pivot, cmap="YlOrRd", vmin=0, vmax=1, cbar_kws={"label": "DT Theft Score"})
    plt.xlabel("Month")
    plt.ylabel("DT Name")
    plt.title(f"DT Theft Scores for Feeder {selected_feeder_short} (January–June 2025)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()
else:
    st.warning(f"No DT data available for feeder {selected_feeder_short}. Check NAME_OF_DT in Customer Data.")

# Heatmap Settings
st.subheader("Customer Heatmap Settings")
num_customers = st.number_input(
    "Number of high-risk customers to display (0 for all)",
    min_value=0,
    max_value=len(customer_monthly[customer_monthly["NAME_OF_DT"] == selected_dt_name]),
    value=10,
    step=1
)

# Customer Heatmap
st.subheader("Theft Analysis")
st.markdown("**Building Theft Probability Heatmap**")
filtered_customers = customer_monthly[customer_monthly["NAME_OF_DT"] == selected_dt_name]
if selected_business_unit != "All":
    filtered_customers = filtered_customers[filtered_customers["BUSINESS_UNIT"] == selected_business_unit]
if selected_undertaking != "All":
    filtered_customers = filtered_customers[filtered_customers["UNDERTAKING"] == selected_undertaking]
if num_customers > 0:
    filtered_customers = filtered_customers.sort_values(by="theft_probability", ascending=False).head(num_customers)
pivot_data = filtered_customers.pivot_table(index="ACCOUNT_NUMBER", columns="month", values="theft_probability", aggfunc="mean")
if not pivot_data.empty:
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data, cmap="YlOrRd", vmin=0, vmax=1, cbar_kws={"label": "Theft Probability"})
    plt.xlabel("Month")
    plt.ylabel("Account Number")
    plt.title(f"Theft Probability for {selected_dt_name} ({selected_feeder_short}, January–June 2025)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()
else:
    st.error("No valid data for customer heatmap. Check ACCOUNT_NUMBER and NAME_OF_DT consistency in 'Debug: Merge inputs for customer_monthly'.")

# Customer List
st.subheader(f"Customers under {selected_dt_name} ({selected_feeder_short}, {selected_month})")
month_customers = filtered_customers[filtered_customers["month"] == selected_month]
if not month_customers.empty:
    styled_df = month_customers[["ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "billed_kwh", 
                                "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", 
                                "Billing_Type", "feeder_score", "dt_score", "meter_status_score", 
                                "account_type_score", "customer_account_type_score", "billing_type_score", 
                                "customer_category_score", "energy_billed_score", "theft_probability", "risk_tier"]].style.format({
        "billed_kwh": "{:.2f}",
        "feeder_score": "{:.3f}",
        "dt_score": "{:.3f}",
        "meter_status_score": "{:.3f}",
        "account_type_score": "{:.3f}",
        "customer_account_type_score": "{:.3f}",
        "billing_type_score": "{:.3f}",
        "customer_category_score": "{:.3f}",
        "energy_billed_score": "{:.3f}",
        "theft_probability": "{:.3f}"
    }).highlight_max(subset=["theft_probability"], color="lightcoral")
    st.dataframe(styled_df)
else:
    st.error("No customers found for the selected DT and month. Check NAME_OF_DT in 'Debug: Merge inputs for customer_monthly'.")

# CSV Export
st.subheader("Export Customer Data")
if not month_customers.empty:
    csv = month_customers[["ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "billed_kwh", 
                           "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", 
                           "Billing_Type", "feeder_score", "dt_score", "meter_status_score", 
                           "account_type_score", "customer_account_type_score", "billing_type_score", 
                           "customer_category_score", "energy_billed_score", "theft_probability", "risk_tier"]].to_csv(index=False)
    st.download_button(
        label=f"Download Customer List ({selected_month})",
        data=csv,
        file_name=f"theft_analysis_{selected_dt_name}_{selected_feeder_short}_{selected_month}.csv",
        mime="text/csv"
    )

# Summary Report
st.subheader("Summary Report")
filtered_dt = dt_merged[dt_merged["New Unique DT Nomenclature"] == selected_dt_name]
for month in months:
    month_data = filtered_dt[filtered_dt["month"] == month]
    if not month_data.empty:
        st.write(f"{month} Energy Lost: {month_data['energy_lost_kwh'].sum():,.2f} kWh")
        st.write(f"{month} Financial Loss: ₦{month_data['financial_loss_naira'].sum():,.2f}")
avg_energy_lost = filtered_dt["energy_lost_kwh"].mean()
avg_financial_loss = filtered_dt["financial_loss_naira"].mean()
st.write(f"Average Monthly Energy Lost: {avg_energy_lost:,.2f} kWh")
st.write(f"Average Monthly Financial Loss: ₦{avg_financial_loss:,.2f}")

# Unbilled Energy Accuracy
st.subheader("Unbilled Energy Accuracy Check")
st.dataframe(dt_df[["New Unique DT Nomenclature", "Provided Avg Monthly Unbilled Energy (kWh)", "Calculated Avg Monthly Unbilled Energy (kWh)", "Unbilled Energy Accuracy"]].style.format({"Provided Avg Monthly Unbilled Energy (kWh)": "{:.2f}", "Calculated Avg Monthly Unbilled Energy (kWh)": "{:.2f}"}))

# Footer
st.markdown("Built by Elvis for Ikeja Electric SIWES III. Field testing version, August 2025.")
st.markdown("Contact: elvisebenuwah@gmail.com | www.linkedin.com/in/elvis-ebenuwah-3956421b2")
