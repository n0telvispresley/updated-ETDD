import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="IE Energy Theft Detection Dashboard", layout="wide")

# File uploader
st.subheader("Upload Excel File")
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.warning("Please upload an Excel file to proceed.")
    st.stop()

# Load Excel file with converters to preserve raw values
try:
    sheets = pd.read_excel(
        uploaded_file,
        sheet_name=None,
        converters={
            "Feeder Data": {"Feeder": str, "Feeder No": str},
            "Transformer Data": {"New Unique DT Nomenclature": str, "DT Number": str},
            "Customer Data_PPM": {"NAME_OF_DT": str, "NAME_OF_FEEDER": str, "ACCOUNT_NUMBER": str, "METER_NUMBER": str},
            "Customer Data_PPD": {"NAME_OF_DT": str, "NAME_OF_FEEDER": str, "ACCOUNT_NUMBER": str, "METER_NUMBER": str},
            "Feeder Band": {"BAND": str, "Feeder": str, "Short Name": str},
            "Customer Tariffs": {"Tariff": str, "Rate (₦)": float}
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

# Combine PPM and PPD into customer_df
ppm_df["Billing_Type"] = "PPM"
ppd_df["Billing_Type"] = "PPD"
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True)

# Data preprocessing
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]

# Convert units
# Feeder Data: MWh to kWh
for month in months:
    feeder_df[month + " (kWh)"] = feeder_df[month] * 1000
feeder_df = feeder_df.drop(columns=months)  # Drop original MWh columns

# Transformer Data: Wh to kWh
for month in months:
    dt_df[month + " (kWh)"] = dt_df[month] / 1000
dt_df = dt_df.drop(columns=months)  # Drop original Wh columns

# Customer Data: kWh for PPM, Wh for PPD
for month in months:
    ppd_df[month + " (kWh)"] = ppd_df[month] / 1000
ppd_df = ppd_df.drop(columns=months)  # Drop original Wh columns

# Extract feeder names from Feeder Data
feeder_names = feeder_df["Feeder"].tolist()

# Map DTs to feeders using prefix matching
def map_dt_to_feeder(dt_name):
    dt_name_str = str(dt_name)
    for feeder in feeder_names:
        if dt_name_str.startswith(feeder):
            return feeder
    return None

dt_df["Feeder"] = dt_df["New Unique DT Nomenclature"].apply(map_dt_to_feeder)
dt_df = dt_df.dropna(subset=["Feeder"])  # Drop DTs with no matching feeder

# Calculate total DT consumption per feeder per month
dt_agg_monthly = dt_df.melt(id_vars=["Feeder"], value_vars=[m + " (kWh)" for m in months], var_name="month", value_name="total_dt_kwh")
dt_agg_monthly["month"] = dt_agg_monthly["month"].str.replace(" (kWh)", "")

# Calculate feeder score per month
feeder_monthly = feeder_df.melt(id_vars=["Feeder"], value_vars=[m + " (kWh)" for m in months], var_name="month", value_name="feeder_energy_kwh")
feeder_monthly["month"] = feeder_monthly["month"].str.replace(" (kWh)", "")
feeder_merged_monthly = feeder_monthly.merge(dt_agg_monthly, on=["Feeder", "month"], how="left")
feeder_merged_monthly["total_dt_kwh"] = feeder_merged_monthly["total_dt_kwh"].fillna(0)
feeder_merged_monthly["feeder_score"] = (1 - feeder_merged_monthly["total_dt_kwh"] / feeder_merged_monthly["feeder_energy_kwh"].replace(0, 1)).clip(0, 1)
feeder_merged_monthly["feeder_energy_lost_kwh"] = feeder_merged_monthly["feeder_energy_kwh"] - feeder_merged_monthly["total_dt_kwh"]
feeder_merged_monthly["feeder_financial_loss_naira"] = feeder_merged_monthly["feeder_energy_lost_kwh"] * 209.5

# Calculate total billed energy per DT per month
customer_monthly = customer_df.melt(id_vars=["NAME_OF_DT", "CUSTOMER_CATEGORY", "BUSINESS_UNIT", "UNDERTAKING"], value_vars=months, var_name="month", value_name="billed_kwh")
customer_agg_monthly = customer_monthly.groupby(["NAME_OF_DT", "month"])["billed_kwh"].sum().reset_index()
customer_agg_monthly.rename(columns={"billed_kwh": "total_billed_kwh"}, inplace=True)

# Calculate DT score and energy lost per month (inverted for theft risk)
dt_monthly = dt_df.melt(id_vars=["New Unique DT Nomenclature"], value_vars=[m + " (kWh)" for m in months], var_name="month", value_name="dt_consumption_kwh")
dt_monthly["month"] = dt_monthly["month"].str.replace(" (kWh)", "")
dt_merged_monthly = dt_monthly.merge(customer_agg_monthly, left_on=["New Unique DT Nomenclature", "month"], right_on=["NAME_OF_DT", "month"], how="left")
dt_merged_monthly["total_billed_kwh"] = dt_merged_monthly["total_billed_kwh"].fillna(0)
dt_merged_monthly["dt_score"] = (1 - dt_merged_monthly["total_billed_kwh"] / dt_merged_monthly["dt_consumption_kwh"].replace(0, 1)).clip(0, 1)
dt_merged_monthly["energy_lost_kwh"] = dt_merged_monthly["dt_consumption_kwh"] - dt_merged_monthly["total_billed_kwh"]
dt_merged_monthly["financial_loss_naira"] = dt_merged_monthly["energy_lost_kwh"] * 209.5

# Verify unbilled energy accuracy in Transformer Data
dt_df["Calculated Avg Monthly Unbilled Energy"] = (dt_df["Consumption (kWh)"] - dt_df["total_billed_kwh"]) / 6
dt_df["Unbilled Energy Accuracy"] = np.abs(dt_df["Calculated Avg Monthly Unbilled Energy"] - dt_df["Avg Monthly Unbilled Energy"]) < 1  # Threshold for accuracy

# Flag inaccurate unbilled energy
st.subheader("Unbilled Energy Accuracy Check")
st.dataframe(dt_df[["New Unique DT Nomenclature", "Avg Monthly Unbilled Energy", "Calculated Avg Monthly Unbilled Energy", "Unbilled Energy Accuracy"]])

# Exclude inactive DTs
dt_df = dt_df[dt_df["Connection Status"] == "Connected"]
dt_df = dt_df[(dt_df[months] > 0).any(axis=1)]  # Flag if inactive with >0 reading (but exclude if 0)

# Streamlit UI
st.title("Ikeja Electric Energy Theft Detection Dashboard")
st.markdown("Detect high-risk buildings using multi-month data (January–June 2025, ₦209.5/kWh).")

# Feeder-Level Summary
st.subheader("Feeder-Level Loss Summary")
feeder_summary = feeder_merged[["Feeder", "feeder_energy_lost_kwh", "feeder_financial_loss_naira"]].sort_values(by="feeder_financial_loss_naira", ascending=False)
st.dataframe(feeder_summary.style.format({"feeder_energy_lost_kwh": "{:,.2f}", "feeder_financial_loss_naira": "₦{:,.2f}"}))

# Filters
col1, col2 = st.columns(2)
with col1:
    feeders = feeder_df["Feeder"].unique()
    selected_feeder = st.selectbox("Select Feeder", feeders)
with col2:
    dt_options = dt_df[dt_df["Feeder"] == selected_feeder]["New Unique DT Nomenclature"].unique()
    selected_dt = st.selectbox("Select DT", dt_options)

# Number of customers to display in heatmap
st.subheader("Heatmap Settings")
num_customers = st.number_input(
    "Number of high-risk customers to display (0 for all)",
    min_value=0,
    max_value=len(filtered_customers),
    value=min(10, len(filtered_customers)),
    step=1
)

# Visuals
st.subheader("Theft Analysis")
st.markdown("**Building Theft Probability Heatmap**")
if num_customers == 0:
    heatmap_data = filtered_customers
else:
    heatmap_data = filtered_customers.head(num_customers)
pivot_data = heatmap_data.pivot(index="ACCOUNT_NUMBER", columns="month", values="theft_probability")
if not pivot_data.empty:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_data, cmap="YlOrRd", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Theft Probability"})
    ax.set_xlabel("Month")
    ax.set_ylabel("Account Number")
    ax.set_title(f"Theft Probability for {selected_dt} ({selected_feeder}, June 2025)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.write("No data available for the selected filters.")

# Customer List
st.subheader(f"Customers under {selected_dt} ({selected_feeder})")
if not filtered_customers.empty:
    st.dataframe(filtered_customers[["ACCOUNT_NUMBER", "CUSTOMER_NAME", "ADDRESS", "ENERGY_BILLED (kWh)", 
                                    "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", 
                                    "feeder_score", "dt_score", "meter_status_score", 
                                    "account_type_score", "customer_account_type_score", 
                                    "energy_billed_score", "theft_probability", "risk_tier"]].style.highlight_max(subset=["theft_probability"], color="lightcoral"))
else:
    st.write("No customers found for the selected filters.")

# Summary Report
st.subheader("Summary Report")
total_energy_lost = filtered_dt["energy_lost_kwh"].sum()
total_financial_loss = filtered_dt["financial_loss_naira"].sum()
st.write(f"Total Energy Lost for {selected_dt} (June 2025): {total_energy_lost:,.2f} kWh")
st.write(f"Total Financial Loss for {selected_dt} (June 2025): ₦{total_financial_loss:,.2f}")
st.write(f"Estimated Yearly Savings per Feeder: ₦{total_financial_loss * 12:,.2f}")

# Footer
st.markdown("Built by Elvis for Ikeja Electric SIWES III. Live demo for August 7, 2025 pitch.")
st.markdown("Contact: elvisebenuwah@gmail.com | www.linkedin.com/in/elvis-ebenuwah-3956421b2")
