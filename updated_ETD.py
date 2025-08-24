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
            "Customer Data_PPM": {"NAME_OF_DT": preserve_exact_string, "NAME_OF_FEEDER": preserve_exact_string, "ACCOUNT_NUMBER": preserve_exact_string, "METER_NUMBER": preserve_exact_string, "BUSINESS_UNIT": preserve_exact_string, "UNDERTAKING": preserve_exact_string, "TARIFF": preserve_exact_string, "CUSTOMER_CATEGORY": preserve_exact_string},
            "Customer Data_PPD": {"NAME_OF_DT": preserve_exact_string, "NAME_OF_FEEDER": preserve_exact_string, "ACCOUNT_NUMBER": preserve_exact_string, "METER_NUMBER": preserve_exact_string, "BUSINESS_UNIT": preserve_exact_string, "UNDERTAKING": preserve_exact_string, "TARIFF": preserve_exact_string, "CUSTOMER_CATEGORY": preserve_exact_string},
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
if feeder_df is None or dt_df is None or ppm_df is None or ppd_df is None or band_df is None or tariff_df is None:
    st.error("One or more sheets (Feeder Data, Transformer Data, Customer Data_PPM, Customer Data_PPD, Feeder Band, Customer Tariffs) not found.")
    st.stop()

# Validate column names
for df, name in [(ppm_df, "Customer Data_PPM"), (ppd_df, "Customer Data_PPD")]:
    if "NAME_OF_DT" not in df.columns:
        st.error(f"Column 'NAME_OF_DT' not found in {name}. Available columns: {df.columns.tolist()}")
        st.stop()
    if "NAME_OF_FEEDER" not in df.columns:
        st.error(f"Column 'NAME_OF_FEEDER' not found in {name}. Available columns: {df.columns.tolist()}")
        st.stop()
if "BAND" not in band_df.columns:
    st.error(f"Column 'BAND' not found in Feeder Band. Available columns: {band_df.columns.tolist()}")
    band_df["BAND"] = ""  # Fallback: empty BAND column
if "Feeder" not in band_df.columns:
    st.error(f"Column 'Feeder' not found in Feeder Band. Available columns: {band_df.columns.tolist()}")
    st.stop()

# Normalize feeder names for consistent matching
feeder_df["Feeder"] = feeder_df["Feeder"].str.strip().str.upper()
ppm_df["NAME_OF_FEEDER"] = ppm_df["NAME_OF_FEEDER"].str.strip().str.upper()
ppd_df["NAME_OF_FEEDER"] = ppd_df["NAME_OF_FEEDER"].str.strip().str.upper()
band_df["Feeder"] = band_df["Feeder"].str.strip().str.upper()

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

# Debug: Show sheet names and column info
if st.checkbox("Show debug info"):
    st.write("Available sheets:", list(sheets.keys()))
    st.write("Customer Data_PPM rows:", len(ppm_df))
    st.write("Customer Data_PPM columns:", ppm_df.columns.tolist())
    st.write("Customer Data_PPD rows:", len(ppd_df))
    st.write("Customer Data_PPD columns:", ppd_df.columns.tolist())
    st.write("Feeder Band rows:", len(band_df))
    st.write("Feeder Band columns:", band_df.columns.tolist())
    st.write("Customer columns:", customer_df.columns.tolist())
    st.write("Sample ACCOUNT_NUMBER values:", customer_df["ACCOUNT_NUMBER"].head().tolist())
    st.write("Count of empty ACCOUNT_NUMBER:", (customer_df["ACCOUNT_NUMBER"] == "").sum())
    st.write("Sample METER_NUMBER values:", customer_df["METER_NUMBER"].head().tolist())
    st.write("Count of empty METER_NUMBER:", (customer_df["METER_NUMBER"] == "").sum())
    st.write("Sample NAME_OF_DT values:", customer_df["NAME_OF_DT"].head().tolist())
    st.write("Sample NAME_OF_FEEDER values:", customer_df["NAME_OF_FEEDER"].head().tolist())
    st.write("Unique NAME_OF_DT values:", sorted(customer_df["NAME_OF_DT"].dropna().astype(str).unique()))
    st.write("Unique NAME_OF_FEEDER values:", sorted(customer_df["NAME_OF_FEEDER"].dropna().astype(str).unique()))
    st.write("Transformer JAN-JUN dtypes:", dt_df[["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]].dtypes)
    st.write("Sample Transformer JAN-JUN values:", dt_df[["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]].head().to_dict())
    st.write("Sample Transformer New Unique DT Nomenclature:", dt_df["New Unique DT Nomenclature"].head().tolist())
    st.write("Sample Feeder Data Feeder values:", feeder_df["Feeder"].head().tolist())
    st.write("Sample Feeder Band Feeder values:", band_df["Feeder"].head().tolist())

# Data preprocessing
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]

# Convert units
# Feeder Data: MWh to kWh
for month in months:
    feeder_df[month + " (kWh)"] = pd.to_numeric(feeder_df[month], errors="coerce") * 1000
    feeder_df[month + " (kWh)"] = feeder_df[month + " (kWh)"].fillna(0)
feeder_df = feeder_df.drop(columns=months)

# Transformer Data: Wh to kWh
for month in months:
    dt_df[month + " (kWh)"] = pd.to_numeric(dt_df[month], errors="coerce") / 1000
    dt_df[month + " (kWh)"] = dt_df[month + " (kWh)"].fillna(0)
dt_df = dt_df.drop(columns=months)

# Customer Data: kWh for PPM, Wh to kWh for PPD
for month in months:
    ppm_df[month + " (kWh)"] = pd.to_numeric(ppm_df[month], errors="coerce").fillna(0)
    ppd_df[month + " (kWh)"] = pd.to_numeric(ppd_df[month], errors="coerce") / 1000
    ppd_df[month + " (kWh)"] = ppd_df[month + " (kWh)"].fillna(0)
ppm_df = ppm_df.drop(columns=months)
ppd_df = ppd_df.drop(columns=months)
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True)

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

# Merge with tariffs
customer_df = customer_df.merge(tariff_df[["Tariff", "Rate (₦)"]], left_on="TARIFF", right_on="Tariff", how="left")
customer_df["Rate (₦)"] = customer_df["Rate (₦)"].fillna(209.5)

# Calculate scores
customer_df["meter_status_score"] = np.where(customer_df["METER_STATUS"] == "Not Metered", 0.9, 0.2)
customer_df["account_type_score"] = np.where(customer_df["ACCOUNT_TYPE"] == "Postpaid", 0.8, 0.3)
customer_df["customer_account_type_score"] = np.where(customer_df["CUSTOMER_ACCOUNT_TYPE"] == "MD", 0.8, 0.3)
customer_df["billing_type_score"] = np.where(customer_df["Billing_Type"] == "PPD", 0.5, 0.2)
customer_df["customer_category_score"] = customer_df["CUSTOMER_CATEGORY"].map({"Residential": 0.2, "Commercial": 0.5, "Special": 0.8}).fillna(0.2)

# Calculate total DT consumption per DT per month
dt_agg = dt_df.melt(id_vars=["New Unique DT Nomenclature", "DT Number", "Flag"], value_vars=[m + " (kWh)" for m in months], var_name="month", value_name="total_dt_kwh")
dt_agg["month"] = dt_agg["month"].str.replace(" (kWh)", "")

# Calculate total billed energy per DT per month
customer_monthly = customer_df.melt(id_vars=["NAME_OF_DT", "ACCOUNT_NUMBER", "Rate (₦)", "CUSTOMER_NAME", "ADDRESS", "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", "CUSTOMER_CATEGORY", "Billing_Type", "NAME_OF_FEEDER", "BUSINESS_UNIT", "UNDERTAKING"], value_vars=[m + " (kWh)" for m in months], var_name="month", value_name="billed_kwh")
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

# Streamlit UI
st.title("Ikeja Electric Energy Theft Detection Dashboard")
st.markdown("Detect high-risk buildings and MD-owned DTs using multi-month data (January–June 2025).")

# Filters
st.subheader("Filters")
col1, col2, col3, col4, col5 = st.columns(5)
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
    if selected_band == "All":
        feeder_options = band_df[band_df["Feeder"].isin(filtered_customer_df["NAME_OF_FEEDER"])].sort_values("BAND")["Short Name"].dropna().astype(str).tolist()
    else:
        feeder_options = band_df[(band_df["BAND"] == selected_band) & (band_df["Feeder"].isin(filtered_customer_df["NAME_OF_FEEDER"]))].sort_values("BAND")["Short Name"].dropna().astype(str).tolist()
    if not feeder_options:
        st.error("No feeders available for the selected band, business unit, or undertaking.")
        st.stop()
    selected_feeder_short = st.selectbox("Select Feeder", feeder_options)
    selected_feeder = band_df[band_df["Short Name"] == selected_feeder_short]["Feeder"].iloc[0] if selected_feeder_short else None
with col5:
    dt_options = filtered_customer_df[filtered_customer_df["NAME_OF_FEEDER"] == selected_feeder]["NAME_OF_DT"].dropna().astype(str).unique().tolist()
    dt_options += [dt for dt in filtered_dt_df[filtered_dt_df["Flag"]]["New Unique DT Nomenclature"].dropna().astype(str).tolist() if dt not in dt_options]
    dt_options = sorted(dt_options)
    if not dt_options:
        st.error(f"No DTs available for feeder {selected_feeder_short}. Check NAME_OF_FEEDER in Customer Data.")
        st.stop()
    dt_options = [f"{dt} (FLAG: Inactive with Energy)" if dt in filtered_dt_df[filtered_dt_df["Flag"]]["New Unique DT Nomenclature"].tolist() else dt for dt in dt_options]
    selected_dt = st.selectbox("Select DT", dt_options)
    selected_dt_name = str(selected_dt).replace(" (FLAG: Inactive with Energy)", "") if selected_dt else ""

# Feeder-Level Summary
st.subheader("Feeder-Level Loss Summary")
feeder_summary = feeder_merged[feeder_merged["month"] == "JUN"].merge(band_df[["Feeder", "Short Name", "BAND"]], on="Feeder", how="left")
feeder_summary["BAND"] = feeder_summary["BAND"].fillna("")  # Fallback for missing BAND
feeder_summary = feeder_summary[feeder_summary["Feeder"].isin(filtered_customer_df["NAME_OF_FEEDER"])]
if feeder_summary.empty:
    st.warning("No feeders match the selected filters. Check NAME_OF_FEEDER in Customer Data and Feeder in Feeder Band.")
else:
    if "BAND" in feeder_summary.columns:
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
