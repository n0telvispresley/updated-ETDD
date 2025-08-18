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

# Load Excel file
try:
    sheets = pd.read_excel(uploaded_file, sheet_name=None)
except Exception as e:
    st.error(f"Error reading Excel file: {e}")
    st.stop()

# Access sheets
feeder_df = sheets.get("Feeder Data")
dt_df = sheets.get("Transformer Data")
customer_df = sheets.get("Customer Data")

# Debug: Show sheet names and column info
if st.checkbox("Show debug info"):
    st.write("Available sheets:", list(sheets.keys()))
    if customer_df is not None:
        st.write("Customer columns:", customer_df.columns.tolist())
        st.write("NAME_OF_DT data type:", customer_df["NAME_OF_DT"].dtype)
        st.write("Sample NAME_OF_DT values:", customer_df["NAME_OF_DT"].head().tolist())
        st.write("METER_NUMBER data type:", customer_df["METER_NUMBER"].dtype)
        st.write("Sample METER_NUMBER values:", customer_df["METER_NUMBER"].head().tolist())
    if dt_df is not None:
        st.write("DT columns:", dt_df.columns.tolist())
        st.write("New Unique DT Nomenclature data type:", dt_df["New Unique DT Nomenclature"].dtype)
        st.write("Sample New Unique DT Nomenclature values:", dt_df["New Unique DT Nomenclature"].head().tolist())

# Check if sheets loaded correctly
if feeder_df is None or dt_df is None or customer_df is None:
    st.error("One or more sheets (Feeder Data, Transformer Data, Customer Data) not found.")
    st.stop()

# Data preprocessing
# Convert Feeder energy to kWh
feeder_df["June Energy (kWh)"] = feeder_df["June Energy (MWh)"] * 1000

# Add month column for heatmap
customer_df["month"] = "June 2025"
dt_df["month"] = "June 2025"

# Ensure consistent data types for merges
customer_df["NAME_OF_DT"] = customer_df["NAME_OF_DT"].astype(str)
dt_df["New Unique DT Nomenclature"] = dt_df["New Unique DT Nomenclature"].astype(str)
dt_df["DT Number"] = dt_df["DT Number"].astype(str)
feeder_df["Feeder"] = feeder_df["Feeder"].astype(str)
customer_df["NAME_OF_FEEDER"] = customer_df["NAME_OF_FEEDER"].astype(str)
customer_df["METER_NUMBER"] = customer_df["METER_NUMBER"].astype(str)

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

# Calculate total DT consumption per feeder
dt_agg = dt_df.groupby("Feeder")["Consumption (kWh)"].sum().reset_index()
dt_agg.rename(columns={"Consumption (kWh)": "total_dt_kwh"}, inplace=True)

# Calculate feeder score (inverted for theft risk)
feeder_merged = feeder_df.merge(dt_agg, on="Feeder", how="left")
feeder_merged["total_dt_kwh"] = feeder_merged["total_dt_kwh"].fillna(0)
feeder_merged["feeder_score"] = (1 - feeder_merged["total_dt_kwh"] / feeder_merged["June Energy (kWh)"].replace(0, 1)).clip(0, 1)
feeder_merged["feeder_energy_lost_kwh"] = feeder_merged["June Energy (kWh)"] - feeder_merged["total_dt_kwh"]
feeder_merged["feeder_financial_loss_naira"] = feeder_merged["feeder_energy_lost_kwh"] * 209.5

# Calculate total billed energy per DT
customer_agg = customer_df.groupby("NAME_OF_DT")["ENERGY_BILLED (kWh)"].sum().reset_index()
customer_agg.rename(columns={"ENERGY_BILLED (kWh)": "total_billed_kwh"}, inplace=True)

# Calculate DT score and energy lost (inverted for theft risk)
dt_merged = dt_df.merge(customer_agg, left_on="New Unique DT Nomenclature", right_on="NAME_OF_DT", how="left")
dt_merged["total_billed_kwh"] = dt_merged["total_billed_kwh"].fillna(0)
dt_merged["dt_score"] = (1 - dt_merged["total_billed_kwh"] / dt_merged["Consumption (kWh)"].replace(0, 1)).clip(0, 1)
dt_merged["energy_lost_kwh"] = dt_merged["Consumption (kWh)"] - dt_merged["total_billed_kwh"]
dt_merged["financial_loss_naira"] = dt_merged["energy_lost_kwh"] * 209.5

# Calculate DT-level theft probability (using dt_score for simplicity)
dt_merged["dt_theft_probability"] = dt_merged["dt_score"]

# Handle MD-owned DTs with no customers
dt_no_customers = dt_merged[~dt_merged["New Unique DT Nomenclature"].isin(customer_df["NAME_OF_DT"])]
md_customers = dt_no_customers[["New Unique DT Nomenclature", "DT Number", "Feeder", "dt_theft_probability", "month"]].copy()
md_customers["METER_NUMBER"] = md_customers["DT Number"]
md_customers["CUSTOMER_NAME"] = md_customers["New Unique DT Nomenclature"]
md_customers["ADDRESS"] = "MD-Owned DT"
md_customers["ENERGY_BILLED (kWh)"] = 0
md_customers["METER_STATUS"] = "Not Metered"
md_customers["ACCOUNT_TYPE"] = "Postpaid"
md_customers["CUSTOMER_ACCOUNT_TYPE"] = "MD"
md_customers["NAME_OF_FEEDER"] = md_customers["Feeder"]
md_customers["theft_probability"] = md_customers["dt_theft_probability"]
md_customers["meter_status_score"] = 0.9  # Not Metered
md_customers["account_type_score"] = 0.8  # Postpaid
md_customers["customer_account_type_score"] = 0.8  # MD
md_customers["energy_billed_score"] = 0.0  # No billed energy

# Append MD-owned DTs to customer_df
customer_df = pd.concat([customer_df, md_customers], ignore_index=True)

# Calculate scores for non-MD customers
customer_df["meter_status_score"] = np.where(customer_df["METER_STATUS"] == "Not Metered", 0.9, 0.2)
customer_df["account_type_score"] = np.where(customer_df["ACCOUNT_TYPE"] == "Postpaid", 0.8, 0.3)
customer_df["customer_account_type_score"] = np.where(customer_df["CUSTOMER_ACCOUNT_TYPE"] == "MD", 0.8, 0.3)
customer_df["energy_billed_score"] = np.where(
    customer_df["CUSTOMER_ACCOUNT_TYPE"] == "MD",
    0.0,
    (1 - customer_df["ENERGY_BILLED (kWh)"] / (1 if customer_df["ENERGY_BILLED (kWh)"].max() == 0 else customer_df["ENERGY_BILLED (kWh)"].max())).clip(0, 1)
)

# Sort DT options by dt_theft_probability (descending) and format with probability
dt_sorted = dt_merged.sort_values(by="dt_theft_probability", ascending=False)
dt_options_sorted = [f"{dt} (Probability: {prob:.2f})" for dt, prob in zip(dt_sorted["New Unique DT Nomenclature"], dt_sorted["dt_theft_probability"])]

# Sort feeders by feeder_financial_loss_naira (descending) and format with lost energy
feeder_merged = feeder_merged.sort_values(by="feeder_financial_loss_naira", ascending=False)
feeder_options_sorted = [f"{feeder} (Lost Energy: {lost:,.2f} kWh)" for feeder, lost in zip(feeder_merged["Feeder"], feeder_merged["feeder_energy_lost_kwh"])]

# Streamlit UI
st.title("Ikeja Electric Energy Theft Detection Dashboard")
st.markdown("Detect high-risk buildings and MD-owned DTs using monthly data (June 2025, ₦209.5/kWh).")

# Feeder-Level Summary
st.subheader("Feeder-Level Loss Summary")
feeder_summary = feeder_merged[["Feeder", "feeder_energy_lost_kwh", "feeder_financial_loss_naira"]].sort_values(by="feeder_financial_loss_naira", ascending=False)
st.dataframe(feeder_summary.style.format({"feeder_energy_lost_kwh": "{:,.2f}", "feeder_financial_loss_naira": "₦{:,.2f}"}))

# Filters
col1, col2 = st.columns(2)
with col1:
    selected_feeder = st.selectbox("Select Feeder", feeder_options_sorted)
    selected_feeder_name = selected_feeder.split(" (Lost Energy:")[0]  # Extract feeder name
with col2:
    dt_options = dt_merged[dt_merged["Feeder"] == selected_feeder_name]["New Unique DT Nomenclature"].unique()
    dt_sorted = dt_merged[dt_merged["Feeder"] == selected_feeder_name].sort_values(by="dt_theft_probability", ascending=False)
    dt_options_sorted = [f"{dt} (Probability: {prob:.2f})" for dt, prob in zip(dt_sorted["New Unique DT Nomenclature"], dt_sorted["dt_theft_probability"])]
    selected_dt = st.selectbox("Select DT", dt_options_sorted)
    selected_dt_name = selected_dt.split(" (Probability:")[0]  # Extract DT name

# Weight Sliders
st.subheader("Theft Probability Weights")
colw1, colw2, colw3 = st.columns(3)
with colw1:
    feeder_weight = st.slider("Feeder Score Weight", 0.0, 1.0, 0.2, 0.05)
    dt_weight = st.slider("DT Score Weight", 0.0, 1.0, 0.3, 0.05)
with colw2:
    meter_weight = st.slider("Meter Status Weight", 0.0, 1.0, 0.2, 0.05)
    account_weight = st.slider("Account Type Weight", 0.0, 1.0, 0.15, 0.05)
with colw3:
    customer_weight = st.slider("Customer Account Type Weight", 0.0, 1.0, 0.15, 0.05)
    energy_weight = st.slider("Energy Billed Weight", 0.0, 1.0, 0.2, 0.05)

# Normalize weights
total_weight = feeder_weight + dt_weight + meter_weight + account_weight + customer_weight + energy_weight
if total_weight == 0:
    st.warning("Total weight cannot be zero. Resetting to default weights.")
    feeder_weight, dt_weight, meter_weight, account_weight, customer_weight, energy_weight = 0.2, 0.3, 0.2, 0.15, 0.15, 0.2
    total_weight = 1.0
feeder_weight /= total_weight
dt_weight /= total_weight
meter_weight /= total_weight
account_weight /= total_weight
customer_weight /= total_weight
energy_weight /= total_weight

# Filter data
filtered_customers = customer_df[customer_df["NAME_OF_DT"] == selected_dt_name]
filtered_dt = dt_merged[dt_merged["New Unique DT Nomenclature"] == selected_dt_name]

# Debug: Check filtered_customers
if st.checkbox("Debug: Show filtered customers info"):
    st.write(f"Filtered customers count: {len(filtered_customers)}")
    st.write("Filtered customers sample:", filtered_customers.head())

# Add feeder_score and dt_score to filtered_customers
filtered_customers = filtered_customers.merge(feeder_merged[["Feeder", "feeder_score"]], 
                                             left_on="NAME_OF_FEEDER", right_on="Feeder", how="left")
filtered_customers = filtered_customers.merge(dt_merged[["New Unique DT Nomenclature", "dt_score"]], 
                                             left_on="NAME_OF_DT", right_on="New Unique DT Nomenclature", how="left")

# Calculate theft probability with weights (use dt_theft_probability for MD-owned DTs)
filtered_customers["theft_probability"] = np.where(
    filtered_customers["CUSTOMER_ACCOUNT_TYPE"] == "MD",
    filtered_customers["theft_probability"].fillna(filtered_customers["dt_score"]),
    (
        feeder_weight * filtered_customers["feeder_score"].fillna(0) +
        dt_weight * filtered_customers["dt_score"].fillna(0) +
        meter_weight * filtered_customers["meter_status_score"] +
        account_weight * filtered_customers["account_type_score"] +
        customer_weight * filtered_customers["customer_account_type_score"] +
        energy_weight * filtered_customers["energy_billed_score"]
    ).clip(0, 1)
)

# Add risk tiers
filtered_customers["risk_tier"] = pd.cut(
    filtered_customers["theft_probability"],
    bins=[0, 0.4, 0.7, 1.0],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

# Sort filtered_customers by theft_probability
filtered_customers = filtered_customers.sort_values(by="theft_probability", ascending=False)

# CSV Export
st.subheader("Export Customer Data")
if not filtered_customers.empty:
    csv = filtered_customers[["METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "ENERGY_BILLED (kWh)", 
                             "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", 
                             "feeder_score", "dt_score", "meter_status_score", 
                             "account_type_score", "customer_account_type_score", 
                             "energy_billed_score", "theft_probability", "risk_tier"]].to_csv(index=False)
    st.download_button(
        label="Download Customer List as CSV",
        data=csv,
        file_name=f"theft_analysis_{selected_dt_name}_{selected_feeder_name}.csv",
        mime="text/csv"
    )

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
if not filtered_customers.empty:
    if num_customers == 0:
        heatmap_data = filtered_customers
    else:
        heatmap_data = filtered_customers.head(num_customers)
    pivot_data = heatmap_data.pivot_table(index="METER_NUMBER", columns="month", values="theft_probability", aggfunc="mean")
    if not pivot_data.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot_data, cmap="YlOrRd", vmin=0, vmax=1, cbar_kws={"label": "Theft Probability"})
        plt.xlabel("Month")
        plt.ylabel("Meter Number")
        plt.title(f"Theft Probability for {selected_dt_name} ({selected_feeder_name}, June 2025)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()  # Clear figure to prevent overlap
    else:
        st.error("No data available for heatmap. Check if customers or MD-owned DT exist for the selected DT.")
else:
    st.error("No customers or MD-owned DT found. Check if NAME_OF_DT in Customer Data matches New Unique DT Nomenclature in Transformer Data.")

# Customer List
st.subheader(f"Customers under {selected_dt_name} ({selected_feeder_name})")
if not filtered_customers.empty:
    styled_df = filtered_customers[["METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "ENERGY_BILLED (kWh)", 
                                   "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", 
                                   "feeder_score", "dt_score", "meter_status_score", 
                                   "account_type_score", "customer_account_type_score", 
                                   "energy_billed_score", "theft_probability", "risk_tier"]].style.format({
        "feeder_score": "{:.3f}",
        "dt_score": "{:.3f}",
        "meter_status_score": "{:.3f}",
        "account_type_score": "{:.3f}",
        "customer_account_type_score": "{:.3f}",
        "energy_billed_score": "{:.3f}",
        "theft_probability": "{:.3f}"
    }).highlight_max(subset=["theft_probability"], color="lightcoral")
    st.dataframe(styled_df)
else:
    st.error("No customers or MD-owned DT found. Check NAME_OF_DT and NAME_OF_FEEDER consistency.")

# Summary Report
st.subheader("Summary Report")
total_energy_lost = filtered_dt[filtered_dt["New Unique DT Nomenclature"] == selected_dt_name]["energy_lost_kwh"].sum()
total_financial_loss = filtered_dt[filtered_dt["New Unique DT Nomenclature"] == selected_dt_name]["financial_loss_naira"].sum()
st.write(f"Total Energy Lost for {selected_dt_name} (June 2025): {total_energy_lost:,.2f} kWh")
st.write(f"Total Financial Loss for {selected_dt_name} (June 2025): ₦{total_financial_loss:,.2f}")
st.write(f"Estimated Yearly Savings per Feeder: ₦{total_financial_loss * 12:,.2f}")

# Footer
st.markdown("Built by Elvis for Ikeja Electric SIWES III. Field testing version, August 2025.")
st.markdown("Contact: elvisebenuwah@gmail.com | www.linkedin.com/in/elvis-ebenuwah-3956421b2")
