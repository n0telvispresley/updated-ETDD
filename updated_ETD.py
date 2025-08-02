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
        st.write("DT_NO data type:", customer_df["DT_NO"].dtype)
        st.write("Sample DT_NO values:", customer_df["DT_NO"].head().tolist())
    if dt_df is not None:
        st.write("DT columns:", dt_df.columns.tolist())

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

# Define feeder names
feeder_names = [
    "11-PTCINJ-T3-Oba Akinjobi",
    "11-PTCINJ-T2-Medical",
    "11-PTCINJ-T3-General Hospital",
    "11-IlupejuINJ-T4-Ikorodu",
    "11-PTCINJ-T1-Olowu",
    "11-Adekunle FajuyiINJ-T1-Isaac John",
    "11-Adekunle FajuyiINJ-T1-Oduduwa",
    "11-MarylandINJ-T1-PTC",
    "11-MarylandINJ-T2-GRA",
    "11-New AlausaINJ-T6-Awolowo",
    "11-New AlausaINJ-T4-Allen",
    "11-PTCINJ-T1-Opebi",
    "11-PTCINJ-T2-Awuse",
    "11-OpebiINJ-T1-Agbaoku",
    "11-OpebiINJ-T1-Salvation",
    "11-IlupejuINJ-T1-ATM",
    "11-IlupejuINJ-T1-Rida Plastic",
    "11-MarylandINJ-T2-Ojota",
    "11-IlupejuINJ-T4-Army Cant."
]

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

# Calculate feeder score
feeder_merged = feeder_df.merge(dt_agg, on="Feeder", how="left")
feeder_merged["total_dt_kwh"] = feeder_merged["total_dt_kwh"].fillna(0)
feeder_merged["feeder_score"] = (feeder_merged["total_dt_kwh"] / feeder_merged["June Energy (kWh)"]).clip(0, 1)
feeder_merged["feeder_energy_lost_kwh"] = feeder_merged["June Energy (kWh)"] - feeder_merged["total_dt_kwh"]
feeder_merged["feeder_financial_loss_naira"] = feeder_merged["feeder_energy_lost_kwh"] * 209.5

# Calculate total billed energy per DT
customer_agg = customer_df.groupby("DT_NO")["ENERGY_BILLED (kWh)"].sum().reset_index()
customer_agg.rename(columns={"ENERGY_BILLED (kWh)": "total_billed_kwh"}, inplace=True)

# Calculate DT score and energy lost
dt_merged = dt_df.merge(customer_agg, left_on="DT Number", right_on="DT_NO", how="left")
dt_merged["total_billed_kwh"] = dt_merged["total_billed_kwh"].fillna(0)
dt_merged["dt_score"] = (dt_merged["total_billed_kwh"] / dt_merged["Consumption (kWh)"]).clip(0, 1)
dt_merged["energy_lost_kwh"] = dt_merged["Consumption (kWh)"] - dt_merged["total_billed_kwh"]
dt_merged["financial_loss_naira"] = dt_merged["energy_lost_kwh"] * 209.5

# Calculate new theft criteria
customer_df["meter_status_score"] = np.where(customer_df["METER_STATUS"] == "Not Metered", 0.9, 0.2)
customer_df["account_type_score"] = np.where(customer_df["ACCOUNT_TYPE"] == "Postpaid", 0.8, 0.3)
customer_df["customer_account_type_score"] = np.where(customer_df["CUSTOMER_ACCOUNT_TYPE"] == "MD", 0.8, 0.3)

# Streamlit UI
st.title("Ikeja Electric Energy Theft Detection Dashboard")
st.markdown("Detect high-risk buildings using monthly data (June 2025, ₦209.5/kWh).")

# Filters
col1, col2 = st.columns(2)
with col1:
    feeders = feeder_df["Feeder"].unique()
    selected_feeder = st.selectbox("Select Feeder", feeders)
with col2:
    dt_options = dt_merged[dt_merged["Feeder"] == selected_feeder]["New Unique DT Nomenclature"].unique()
    selected_dt = st.selectbox("Select DT", dt_options)

# Filter data
dt_number = dt_merged[dt_merged["New Unique DT Nomenclature"] == selected_dt]["DT Number"].iloc[0]
filtered_customers = customer_df[customer_df["DT_NO"] == dt_number]
filtered_dt = dt_merged[dt_merged["Feeder"] == selected_feeder]

# Add feeder_score and dt_score to filtered_customers
filtered_customers = filtered_customers.merge(feeder_merged[["Feeder", "feeder_score"]], 
                                             left_on="NAME_OF_FEEDER", right_on="Feeder", how="left")
filtered_customers = filtered_customers.merge(dt_merged[["DT Number", "dt_score"]], 
                                             left_on="DT_NO", right_on="DT Number", how="left")

# Calculate energy billed score
max_billed_kwh = filtered_customers["ENERGY_BILLED (kWh)"].max()
filtered_customers["energy_billed_score"] = (1 - filtered_customers["ENERGY_BILLED (kWh)"] / (1 if max_billed_kwh == 0 else max_billed_kwh)).clip(0, 1)

# Calculate theft probability with weights (using inverse of scores for higher theft risk)
filtered_customers["theft_probability"] = (
    0.2 * (1 - filtered_customers["feeder_score"]) +
    0.3 * (1 - filtered_customers["dt_score"]) +
    0.2 * filtered_customers["meter_status_score"] +
    0.15 * filtered_customers["account_type_score"] +
    0.15 * filtered_customers["customer_account_type_score"] +
    0.2 * filtered_customers["energy_billed_score"]
).clip(0, 1)

# Visuals
st.subheader("Theft Analysis")
col3, col4 = st.columns(2)

# Building Theft Probability Heatmap (All Customers)
with col3:
    st.markdown("**Building Theft Probability Heatmap (All Customers)**")
    pivot_data = filtered_customers.pivot(index="ACCOUNT_NUMBER", columns="month", values="theft_probability")
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

# Top 10 High-Risk Buildings Heatmap
with col4:
    st.markdown("**Top 10 High-Risk Buildings Heatmap**")
    top_10_customers = filtered_customers.nlargest(10, "theft_probability")
    pivot_top_10 = top_10_customers.pivot(index="ACCOUNT_NUMBER", columns="month", values="theft_probability")
    if not pivot_top_10.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_top_10, cmap="YlOrRd", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Theft Probability"})
        ax.set_xlabel("Month")
        ax.set_ylabel("Account Number")
        ax.set_title(f"Top 10 High-Risk Buildings for {selected_dt} ({selected_feeder}, June 2025)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No data available for top 10 high-risk buildings.")

# Customer List
st.subheader(f"Customers under {selected_dt} ({selected_feeder})")
if not filtered_customers.empty:
    st.dataframe(filtered_customers[["ACCOUNT_NUMBER", "CUSTOMER_NAME", "ADDRESS", "ENERGY_BILLED (kWh)", 
                                    "METER_STATUS", "ACCOUNT_TYPE", "CUSTOMER_ACCOUNT_TYPE", 
                                    "feeder_score", "dt_score", "meter_status_score", 
                                    "account_type_score", "customer_account_type_score", 
                                    "energy_billed_score", "theft_probability"]]
                 .style.highlight_max(subset=["theft_probability"], color="lightcoral"))
else:
    st.write("No customers found for the selected filters.")

# Summary Report
st.subheader("Summary Report")
total_energy_lost = filtered_dt[filtered_dt["DT Number"] == dt_number]["energy_lost_kwh"].sum()
total_financial_loss = filtered_dt[filtered_dt["DT Number"] == dt_number]["financial_loss_naira"].sum()
st.write(f"Total Energy Lost for {selected_dt} (June 2025): {total_energy_lost:,.2f} kWh")
st.write(f"Total Financial Loss for {selected_dt} (June 2025): ₦{total_financial_loss:,.2f}")
st.write(f"Estimated Yearly Savings per Feeder: ₦{total_financial_loss * 12:,.2f}")
st.write(f"Total Yearly Savings (4 Feeders): ₦{total_financial_loss * 12 * 4:,.2f}")

# Footer
st.markdown("Built by Elvis for Ikeja Electric SIWES III. Live demo for August 7, 2025 pitch.")
st.markdown("Contact: [Your LinkedIn] | [your-app.streamlit.app]")
