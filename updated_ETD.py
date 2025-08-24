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
def get_short_name(name):
    if isinstance(name, str) and name and "-" in name:
        return name.split("-")[-1].strip()
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
            "Feeder Data": {"Feeder": preserve_exact_string, "Feeder No": preserve_exact_string},
            "Transformer Data": {"New Unique DT Nomenclature": preserve_exact_string, "DT Number": preserve_exact_string, "Ownership": preserve_exact_string, "Connection Status": preserve_exact_string, "UNDERTAKING": preserve_exact_string},
            "Customer Data_PPM": {"NAME_OF_DT": preserve_exact_string, "NAME_OF_FEEDER": preserve_exact_string, "ACCOUNT_NUMBER": preserve_exact_string, "METER_NUMBER": preserve_exact_string, "BUSINESS_UNIT": preserve_exact_string, "UNDERTAKING": preserve_exact_string, "TARIFF": preserve_exact_string, "CUSTOMER_CATEGORY": preserve_exact_string, "METER_STATUS": preserve_exact_string, "ACCOUNT_TYPE": preserve_exact_string, "CUSTOMER_ACCOUNT_TYPE": preserve_exact_string, "DT_NO": preserve_exact_string, "FEEDER_NO": preserve_exact_string},
            "Customer Data_PPD": {"NAME_OF_DT": preserve_exact_string, "NAME_OF_FEEDER": preserve_exact_string, "ACCOUNT_NUMBER": preserve_exact_string, "METER_NUMBER": preserve_exact_string, "BUSINESS_UNIT": preserve_exact_string, "UNDERTAKING": preserve_exact_string, "TARIFF": preserve_exact_string, "CUSTOMER_CATEGORY": preserve_exact_string, "METER_STATUS": preserve_exact_string, "ACCOUNT_TYPE": preserve_exact_string, "CUSTOMER_ACCOUNT_TYPE": preserve_exact_string, "DT_NO": preserve_exact_string, "FEEDER_NO": preserve_exact_string},
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
required_customer_cols = ["NAME_OF_DT", "ACCOUNT_NUMBER", "CUSTOMER_NAME", "ADDRESS", "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", "NAME_OF_FEEDER", "BUSINESS_UNIT", "UNDERTAKING"]
required_dt_cols = ["New Unique DT Nomenclature", "DT Number"]
required_feeder_cols = ["Feeder"]
for df, name, cols in [(ppm_df, "Customer Data_PPM", required_customer_cols), (ppd_df, "Customer Data_PPD", required_customer_cols), (dt_df, "Transformer Data", required_dt_cols), (feeder_df, "Feeder Data", required_feeder_cols)]:
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in {name}: {missing_cols}")
        st.stop()

# Handle missing columns
for df, col in [(ppm_df, "TARIFF"), (ppd_df, "TARIFF"), (ppm_df, "DT_NO"), (ppd_df, "DT_NO"), (band_df, "BAND"), (tariff_df, "Tariff")]:
    if col not in df.columns:
        st.warning(f"Column '{col}' not found in {df.name if hasattr(df, 'name') else 'sheet'}. Creating empty column.")
        df[col] = ""
if "Feeder" not in band_df.columns:
    st.error(f"Column 'Feeder' not found in Feeder Band.")
    st.stop()
if "Short Name" not in band_df.columns:
    band_df["Short Name"] = band_df["Feeder"].apply(get_short_name)

# Handle Rate column variants
rate_variants = ["Rate (NGN)", "Rate (₦)", "Rate", "RATE", "Rate(NGN)", "Rate(₦)"]
rate_col = next((col for col in rate_variants if col in tariff_df.columns), None)
if rate_col:
    tariff_df["Rate (NGN)"] = pd.to_numeric(tariff_df[rate_col], errors="coerce").fillna(209.5)
    if rate_col != "Rate (NGN)":
        tariff_df = tariff_df.drop(columns=[rate_col], errors="ignore")
else:
    tariff_df["Rate (NGN)"] = 209.5

# Normalize names
for col, df in [
    ("Feeder", feeder_df), ("NAME_OF_FEEDER", ppm_df), ("NAME_OF_FEEDER", ppd_df), ("Feeder", band_df),
    ("NAME_OF_DT", ppm_df), ("NAME_OF_DT", ppd_df), ("New Unique DT Nomenclature", dt_df),
    ("TARIFF", ppm_df), ("TARIFF", ppd_df), ("Tariff", tariff_df),
    ("DT_NO", ppm_df), ("DT_NO", ppd_df), ("DT Number", dt_df),
    ("FEEDER_NO", ppm_df), ("FEEDER_NO", ppd_df), ("Feeder No", feeder_df)
]:
    df[col] = df[col].astype(str).str.strip().str.upper()

# Handle missing UNDERTAKING
for df in [ppm_df, ppd_df]:
    df["UNDERTAKING"] = df["UNDERTAKING"].astype(str).fillna("PTC").replace(["", "N/A"], "PTC")
if "UNDERTAKING" not in dt_df.columns:
    dt_df["UNDERTAKING"] = "PTC"
else:
    dt_df["UNDERTAKING"] = dt_df["UNDERTAKING"].astype(str).fillna("PTC").replace(["", "N/A"], "PTC")

# Combine PPM and PPD
ppm_df["Billing_Type"] = "PPM"
ppd_df["Billing_Type"] = "PPD"
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True)
if customer_df.empty:
    st.error("customer_df is empty.")
    st.stop()

# Create short names
feeder_df["Feeder_Short"] = feeder_df["Feeder"].apply(get_short_name)
dt_df["DT_Short_Name"] = dt_df["New Unique DT Nomenclature"].apply(get_short_name)
dt_df["Feeder_Short"] = dt_df["New Unique DT Nomenclature"].apply(get_short_name)
customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(get_short_name)
customer_df["Feeder_Short"] = customer_df["NAME_OF_FEEDER"].apply(get_short_name)

# Map customers to DTs
customer_df = customer_df.merge(
    dt_df[["DT Number", "New Unique DT Nomenclature", "Feeder_Short", "DT_Short_Name", "UNDERTAKING"]],
    left_on="DT_NO",
    right_on="DT Number",
    how="left"
)
customer_df["NAME_OF_DT"] = customer_df["New Unique DT Nomenclature"].combine_first(customer_df["NAME_OF_DT"])
customer_df["NAME_OF_FEEDER"] = customer_df["Feeder_Short"].combine_first(customer_df["Feeder_Short"])
customer_df["UNDERTAKING"] = customer_df["UNDERTAKING_y"].combine_first(customer_df["UNDERTAKING_x"]).fillna("PTC")
customer_df = customer_df.drop(columns=["DT Number", "New Unique DT Nomenclature", "Feeder_Short", "DT_Short_Name", "UNDERTAKING_x", "UNDERTAKING_y"], errors="ignore")

# Merge tariffs
customer_df = customer_df.merge(tariff_df[["Tariff", "Rate (NGN)"]], left_on="TARIFF", right_on="Tariff", how="left")
customer_df["Rate (NGN)"] = customer_df["Rate (NGN)"].fillna(209.5)
customer_df = customer_df.drop(columns=["Tariff"], errors="ignore")

# Debug
if st.checkbox("Debug: Data"):
    st.write("customer_df TARIFF:", sorted(customer_df["TARIFF"].dropna().astype(str).unique()))
    st.write("tariff_df Tariff:", sorted(tariff_df["Tariff"].dropna().astype(str).unique()))
    st.write("customer_df DT_NO:", customer_df["DT_NO"].head().tolist())
    st.write("customer_df NAME_OF_DT:", customer_df["NAME_OF_DT"].head().tolist())
    st.write("customer_df Feeder_Short:", customer_df["Feeder_Short"].head().tolist())
    st.write("customer_df UNDERTAKING:", customer_df["UNDERTAKING"].head().tolist())
    st.write("dt_df DT Number:", sorted(dt_df["DT Number"].dropna().astype(str).unique()))
    st.write("dt_df DT_Short_Name:", sorted(dt_df["DT_Short_Name"].dropna().astype(str).unique()))
    st.write("dt_df Feeder_Short:", sorted(dt_df["Feeder_Short"].dropna().astype(str).unique()))

# Data preprocessing
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
for month in months:
    for df, unit in [(feeder_df, 1000), (ppm_df, 1), (ppd_df, 0.001)]:
        col = f"{month} (kWh)"
        if month in df.columns:
            df[col] = pd.to_numeric(df[month], errors="coerce").fillna(0) * unit
        else:
            df[col] = 0
    if month in dt_df.columns:
        dt_df[f"{month} (kWh)"] = pd.to_numeric(dt_df[month], errors="coerce").fillna(0) / 1000
    else:
        dt_df[f"{month} (kWh)"] = 0
for df in [feeder_df, dt_df, ppm_df, ppd_df]:
    df.drop(columns=months, errors="ignore", inplace=True)

# Recombine customer_df
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True)
customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(get_short_name)
customer_df["Feeder_Short"] = customer_df["NAME_OF_FEEDER"].apply(get_short_name)

# Melt customer data
required_id_vars = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "CUSTOMER_NAME", "ADDRESS", "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", "Billing_Type", "Feeder_Short", "BUSINESS_UNIT", "UNDERTAKING", "DT_NO"]
if "Rate (NGN)" in customer_df.columns:
    required_id_vars.append("Rate (NGN)")
value_vars = [f"{m} (kWh)" for m in months]
try:
    customer_monthly = customer_df.melt(id_vars=required_id_vars, value_vars=value_vars, var_name="month", value_name="billed_kwh")
    customer_monthly["month"] = customer_monthly["month"].str.replace(" (kWh)", "")
except Exception as e:
    st.error(f"Melt failed: {e}")
    st.stop()

# Unbilled energy accuracy
dt_df["Calculated Avg Monthly Unbilled Energy (kWh)"] = sum(dt_df[f"{m} (kWh)"] for m in months) / 6
dt_df["Provided Avg Monthly Unbilled Energy (kWh)"] = pd.to_numeric(dt_df.get("Avg Monthly Unbilled Energy", 0), errors="coerce").fillna(0) / 1000
dt_df["Unbilled Energy Discrepancy (kWh)"] = np.abs(dt_df["Calculated Avg Monthly Unbilled Energy (kWh)"] - dt_df["Provided Avg Monthly Unbilled Energy (kWh)"])
dt_df["Unbilled Energy Accuracy"] = dt_df["Unbilled Energy Discrepancy (kWh)"] < 1

# Filter inactive DTs
dt_df["Has Energy"] = dt_df[[f"{m} (kWh)" for m in months]].gt(0).any(axis=1)
dt_df["Flag"] = (dt_df["Connection Status"] == "Not Connected") & (dt_df["Has Energy"])
dt_df = dt_df[(dt_df["Connection Status"] == "Connected") | dt_df["Flag"]]

# Handle private DTs
dt_no_customers = dt_df[~dt_df["DT Number"].isin(customer_df["DT_NO"])]
private_dts = dt_no_customers[["New Unique DT Nomenclature", "DT Number", "Feeder_Short", "DT_Short_Name", "UNDERTAKING"]].copy()
private_dts["ACCOUNT_NUMBER"] = private_dts["DT Number"]
private_dts["METER_NUMBER"] = private_dts["DT Number"]
private_dts["CUSTOMER_NAME"] = private_dts["DT_Short_Name"] + " (Private DT)"
private_dts["ADDRESS"] = "Private DT"
private_dts["METER_STATUS"] = "Metered"
private_dts["ACCOUNT_TYPE"] = "Postpaid"
private_dts["CUSTOMER_ACCOUNT_TYPE"] = "MD"
private_dts["Billing_Type"] = "PPD"
private_dts["CUSTOMER_CATEGORY"] = "Special"
private_dts["Rate (NGN)"] = 209.5
private_dts["NAME_OF_FEEDER"] = private_dts["Feeder_Short"]
private_dts["Feeder_Short"] = private_dts["Feeder_Short"]
private_dts["BUSINESS_UNIT"] = "PTC"
private_dts["UNDERTAKING"] = private_dts["UNDERTAKING"].fillna("PTC")
private_dts["DT_NO"] = private_dts["DT Number"]
for month in months:
    private_dts[f"{month} (kWh)"] = dt_no_customers[f"{month} (kWh)"]
private_dts = private_dts.drop(columns=["Feeder_Short"], errors="ignore")
customer_df = pd.concat([customer_df, private_dts], ignore_index=True)

# Recalculate customer_monthly
customer_monthly = customer_df.melt(id_vars=required_id_vars, value_vars=value_vars, var_name="month", value_name="billed_kwh")
customer_monthly["month"] = customer_monthly["month"].str.replace(" (kWh)", "")

# Calculate scores
customer_df["meter_status_score"] = np.where(customer_df["METER_STATUS"] == "Not Metered", 0.9, 0.2)
customer_df["account_type_score"] = np.where(customer_df["ACCOUNT_TYPE"] == "Postpaid", 0.8, 0.3)
customer_df["customer_account_type_score"] = np.where(customer_df["CUSTOMER_ACCOUNT_TYPE"] == "MD", 0.8, 0.3)
customer_df["billing_type_score"] = np.where(customer_df["Billing_Type"] == "PPD", 0.5, 0.2)
customer_df["customer_category_score"] = customer_df["CUSTOMER_CATEGORY"].map({"Residential": 0.2, "Commercial": 0.5, "Special": 0.8}).fillna(0.2)

# DT consumption
dt_agg = dt_df.melt(id_vars=["New Unique DT Nomenclature", "DT Number", "Flag", "DT_Short_Name", "Feeder_Short"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="total_dt_kwh")
dt_agg["month"] = dt_agg["month"].str.replace(" (kWh)", "")

# Billed energy
customer_agg = customer_monthly.groupby(["DT_NO", "month"])["billed_kwh"].sum().reset_index()
customer_agg.rename(columns={"billed_kwh": "total_billed_kwh"}, inplace=True)

# DT scores
dt_merged = dt_agg.merge(customer_agg, left_on=["DT Number", "month"], right_on=["DT_NO", "month"], how="left")
dt_merged["total_billed_kwh"] = dt_merged["total_billed_kwh"].fillna(0)
dt_merged["dt_score"] = np.where(
    dt_merged["New Unique DT Nomenclature"].isin(dt_no_customers["New Unique DT Nomenclature"]),
    0,
    (1 - dt_merged["total_billed_kwh"] / dt_merged["total_dt_kwh"].replace(0, 1)).clip(0, 1)
)
dt_merged["energy_lost_kwh"] = dt_merged["total_dt_kwh"] - dt_merged["total_billed_kwh"]
dt_merged["financial_loss_naira"] = dt_merged["energy_lost_kwh"] * 209.5

# Feeder scores
feeder_monthly = feeder_df.melt(id_vars=["Feeder", "Feeder_Short"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="feeder_energy_kwh")
feeder_monthly["month"] = feeder_monthly["month"].str.replace(" (kWh)", "")
feeder_agg = customer_monthly.groupby(["Feeder_Short", "month"])["billed_kwh"].sum().reset_index().rename(columns={"billed_kwh": "total_billed_kwh"})
feeder_merged = feeder_monthly.merge(feeder_agg, left_on=["Feeder_Short", "month"], right_on=["Feeder_Short", "month"], how="left")
feeder_merged["total_billed_kwh"] = feeder_merged["total_billed_kwh"].fillna(0)
feeder_merged["feeder_score"] = (1 - feeder_merged["total_billed_kwh"] / feeder_merged["feeder_energy_kwh"].replace(0, 1)).clip(0, 1)
feeder_merged["feeder_energy_lost_kwh"] = feeder_merged["feeder_energy_kwh"] - feeder_merged["total_billed_kwh"]
feeder_merged = feeder_merged.merge(band_df[["Feeder", "Short Name", "BAND"]], on="Feeder", how="left")
feeder_merged["BAND"] = feeder_merged["BAND"].fillna("Unknown")
feeder_merged["Short Name"] = feeder_merged["Short Name"].fillna(feeder_merged["Feeder_Short"])
feeder_merged["feeder_financial_loss_naira"] = feeder_merged["feeder_energy_lost_kwh"] * 209.5

# Streamlit UI
st.title("Ikeja Electric Energy Theft Detection Dashboard")
st.markdown("Detect high-risk buildings and private DTs (January–June 2025).")

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
with col4:
    feeder_options = sorted(feeder_df["Feeder_Short"].dropna().astype(str).unique())
    selected_feeder_short = st.selectbox("Select Feeder", feeder_options)
    feeder_full = feeder_df[feeder_df["Feeder_Short"] == selected_feeder_short]["Feeder"].iloc[0] if not feeder_df[feeder_df["Feeder_Short"] == selected_feeder_short].empty else selected_feeder_short
with col5:
    dt_options = dt_df[dt_df["Feeder_Short"] == selected_feeder_short]["DT_Short_Name"].dropna().astype(str).unique().tolist()
    dt_options = sorted(dt_options)
    if not dt_options:
        st.error(f"No DTs available for feeder {selected_feeder_short}. Check Feeder and New Unique DT Nomenclature in Transformer Data.")
        st.write("Available Feeder_Short values:", sorted(dt_df["Feeder_Short"].dropna().astype(str).unique()))
        st.stop()
    dt_short_to_full = {dt: dt_df[dt_df["DT_Short_Name"] == dt]["New Unique DT Nomenclature"].iloc[0] for dt in dt_options}
    dt_options_display = [f"{dt} (Inactive with Energy)" if dt_short_to_full[dt] in dt_df[dt_df["Flag"]]["New Unique DT Nomenclature"].tolist() else dt for dt in dt_options]
    selected_dt_short = st.selectbox("Select DT", dt_options_display)
    selected_dt_name = dt_short_to_full.get(selected_dt_short.replace(" (Inactive with Energy)", ""), "")
with col6:
    month_options = ["All"] + months
    selected_month = st.selectbox("Select Month", month_options)

# Apply filters
filtered_customer_df = customer_df.copy()
if selected_business_unit != "All":
    filtered_customer_df = filtered_customer_df[filtered_customer_df["BUSINESS_UNIT"] == selected_business_unit]
if selected_undertaking != "All":
    filtered_customer_df = filtered_customer_df[filtered_customer_df["UNDERTAKING"] == selected_undertaking]
filtered_dt_df = dt_df[dt_df["Feeder_Short"] == selected_feeder_short]

# DT Theft Probability Heatmap
st.subheader("DT Theft Probability Heatmap")
filtered_dt_agg = dt_agg[dt_agg["Feeder_Short"] == selected_feeder_short]
if filtered_dt_agg.empty:
    st.error(f"No DT data for feeder {selected_feeder_short}. Check Transformer Data.")
    st.stop()
pivot_values = "dt_score" if "dt_score" in filtered_dt_agg.columns else "total_dt_kwh"
dt_pivot = filtered_dt_agg.pivot_table(index="DT_Short_Name", columns="month", values=pivot_values, aggfunc="mean")
if not dt_pivot.empty:
    plt.figure(figsize=(10, 8))
    sns.heatmap(dt_pivot, cmap="YlOrRd", cbar_kws={"label": "DT Theft Score" if pivot_values == "dt_score" else "Total DT kWh"})
    plt.xlabel("Month")
    plt.ylabel("DT Name")
    plt.title(f"DT {'Theft Score' if pivot_values == 'dt_score' else 'Energy Consumption'} for {selected_feeder_short}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()
else:
    st.error(f"No DT data for {selected_feeder_short} after pivoting.")
    st.stop()

# Customer scores
customer_monthly = customer_monthly.merge(feeder_merged[["Feeder_Short", "month", "feeder_score"]], left_on=["Feeder_Short", "month"], right_on=["Feeder_Short", "month"], how="left")
customer_monthly = customer_monthly.merge(dt_merged[["DT Number", "month", "dt_score"]], left_on=["DT_NO", "month"], right_on=["DT Number", "month"], how="left")
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
customer_monthly["theft_probability"] = np.where(
    customer_monthly["NAME_OF_DT"].isin(dt_no_customers["New Unique DT Nomenclature"]),
    0,
    customer_monthly["theft_probability"]
)
customer_monthly["risk_tier"] = pd.cut(customer_monthly["theft_probability"], bins=[0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"], include_lowest=True)

# Feeder-Level Loss Summary
st.subheader("Feeder-Level Loss Summary")
feeder_summary = feeder_merged.merge(band_df[["Feeder", "Short Name", "BAND"]], on="Feeder", how="left")
feeder_summary["BAND"] = feeder_summary["BAND"].fillna("Unknown")
feeder_summary["Short Name"] = feeder_summary["Short Name"].fillna(feeder_summary["Feeder_Short"])
feeder_summary = feeder_summary[feeder_summary["Feeder_Short"].isin(feeder_df["Feeder_Short"])]
if feeder_summary.empty:
    st.error("No feeders match filters.")
    st.stop()
feeder_pivot = feeder_summary.pivot_table(index=["Short Name", "BAND"], columns="month", values=["feeder_energy_lost_kwh", "feeder_financial_loss_naira"], aggfunc="sum").fillna(0)
st.dataframe(feeder_pivot.style.format("{:,.2f}"))

# Customer Heatmap Settings
st.subheader("Customer Heatmap Settings")
num_customers = st.number_input("Number of high-risk customers (0 for all)", min_value=0, max_value=len(customer_monthly[customer_monthly["DT_Short_Name"] == selected_dt_short]), value=10, step=1)

# Customer Heatmap
st.subheader("Theft Analysis")
st.markdown("**Building Theft Probability Heatmap**")
filtered_customers = customer_monthly[customer_monthly["DT_Short_Name"] == selected_dt_short]
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
    plt.title(f"Theft Probability for {selected_dt_short} ({selected_feeder_short})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()
else:
    st.warning(f"No customer data for {selected_dt_short}. Private DTs have theft_probability = 0.")

# Customer List
st.subheader(f"Customers under {selected_dt_short} ({selected_feeder_short}, {selected_month})")
if selected_month == "All":
    month_customers = filtered_customers.groupby(["ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", "Billing_Type"]).agg({
        "billed_kwh": "sum",
        "feeder_score": "mean",
        "dt_score": "mean",
        "meter_status_score": "mean",
        "account_type_score": "mean",
        "customer_account_type_score": "mean",
        "billing_type_score": "mean",
        "customer_category_score": "mean",
        "energy_billed_score": "mean",
        "theft_probability": "mean",
        "risk_tier": lambda x: pd.Series(x).mode()[0]
    }).reset_index()
else:
    month_customers = filtered_customers[filtered_customers["month"] == selected_month]
if not month_customers.empty:
    styled_df = month_customers[["ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "billed_kwh", "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", "Billing_Type", "feeder_score", "dt_score", "meter_status_score", "account_type_score", "customer_account_type_score", "billing_type_score", "customer_category_score", "energy_billed_score", "theft_probability", "risk_tier"]].style.format({
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
    st.warning(f"No customers for {selected_dt_short} ({selected_month}).")

# CSV Export
st.subheader("Export Customer Data")
if not month_customers.empty:
    csv = month_customers[["ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "billed_kwh", "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", "Billing_Type", "feeder_score", "dt_score", "meter_status_score", "account_type_score", "customer_account_type_score", "billing_type_score", "customer_category_score", "energy_billed_score", "theft_probability", "risk_tier"]].to_csv(index=False)
    st.download_button(label=f"Download Customer List ({selected_month})", data=csv, file_name=f"theft_analysis_{selected_dt_short}_{selected_feeder_short}_{selected_month}.csv", mime="text/csv")

# Summary Report
st.subheader("Summary Report")
filtered_dt = dt_merged[dt_merged["DT_Short_Name"] == selected_dt_short]
summary_data = filtered_dt.pivot_table(index="DT_Short_Name", columns="month", values=["energy_lost_kwh", "financial_loss_naira"], aggfunc="sum").fillna(0)
if not summary_data.empty:
    st.dataframe(summary_data.style.format("{:,.2f}"))
    avg_energy_lost = filtered_dt["energy_lost_kwh"].mean()
    avg_financial_loss = filtered_dt["financial_loss_naira"].mean()
    st.write(f"Average Monthly Energy Lost: {avg_energy_lost:,.2f} kWh")
    st.write(f"Average Monthly Financial Loss: ₦{avg_financial_loss:,.2f}")
else:
    st.error(f"No data for {selected_dt_short}.")

# Unbilled Energy Accuracy
st.subheader("Unbilled Energy Accuracy Check")
st.dataframe(dt_df[["DT_Short_Name", "Provided Avg Monthly Unbilled Energy (kWh)", "Calculated Avg Monthly Unbilled Energy (kWh)", "Unbilled Energy Accuracy"]].style.format({"Provided Avg Monthly Unbilled Energy (kWh)": "{:.2f}", "Calculated Avg Monthly Unbilled Energy (kWh)": "{:.2f}"}))

# Footer
st.markdown("Built by Elvis for Ikeja Electric SIWES III. August 2025.")
st.markdown("Contact: elvisebenuwah@gmail.com | www.linkedin.com/in/elvis-ebenuwah-3956421b2")
