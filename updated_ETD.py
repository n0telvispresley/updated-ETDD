import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Streamlit page config
st.set_page_config(page_title="IE Energy Theft Detection Dashboard (ML)", layout="wide")

# --- Utility Functions (Kept as is) ---
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

def add_feeder_column_safe(df, name_of_dt_col="NAME_OF_DT"):
    if name_of_dt_col not in df.columns:
        st.error(f"Column '{name_of_dt_col}' missing. Cannot derive 'Feeder'.")
        df["Feeder"] = ""
        return df
    df = df.copy()
    df["Feeder"] = df[name_of_dt_col].apply(
        lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x and len(x.split("-")) >= 3 else x
    )
    df["Feeder"] = df["Feeder"].apply(normalize_name)
    return df

# --- Feature Calculation Functions (Kept as is) ---

def calculate_pattern_deviation(df, id_col, value_cols):
    results = []
    valid_cols = [c for c in value_cols if c in df.columns]
    if not valid_cols:
        return pd.DataFrame({"id": [], "pattern_deviation_score": []})
    for id_val, group in df.groupby(id_col):
        values = group[valid_cols].iloc[0].values.astype(float)
        nonzero = values[values > 0]
        if len(nonzero) == 0:
            score = 1.0
        else:
            max_nonzero = nonzero.max()
            below = np.sum(values < 0.6 * max_nonzero)
            score = below / len(valid_cols)
        results.append({"id": id_val, "pattern_deviation_score": min(score, 1.0)})
    return pd.DataFrame(results).rename(columns={"id": id_col})

def calculate_zero_counter(df, id_col, value_cols):
    results = []
    valid_cols = [c for c in value_cols if c in df.columns]
    if not valid_cols:
        return pd.DataFrame({"id": [], "zero_counter_score": []})
    for id_val, group in df.groupby(id_col):
        values = group[valid_cols].iloc[0].values.astype(float)
        zeros = np.sum(values == 0)
        score = zeros / len(valid_cols) if len(valid_cols) > 0 else 0.0
        results.append({"id": id_val, "zero_counter_score": min(score, 1.0)})
    return pd.DataFrame(results).rename(columns={"id": id_col})

def calculate_dt_relative_usage(customer_monthly):
    cust_sum = customer_monthly.groupby(["ACCOUNT_NUMBER", "NAME_OF_DT"], as_index=False)["billed_kwh"].sum()
    dt_avg = cust_sum[cust_sum["billed_kwh"] > 0].groupby("NAME_OF_DT", as_index=False)["billed_kwh"].mean().rename(columns={"billed_kwh": "dt_avg_kwh"})
    cust_sum = cust_sum.merge(dt_avg, on="NAME_OF_DT", how="left")
    
    # Simple logic to score low consumption relative to DT average as high risk
    def _score(row):
        if pd.isna(row["dt_avg_kwh"]) or row["dt_avg_kwh"] == 0:
            return 0.5 if row["billed_kwh"] == 0 else 0.1
        if row["billed_kwh"] < 0.3 * row["dt_avg_kwh"]: # Consumption is less than 30% of average DT usage -> High Risk
            return 0.9
        if row["billed_kwh"] > 0.7 * row["dt_avg_kwh"]: # Consumption is more than 70% of average DT usage -> Low Risk
            return 0.1
        # Interpolate between 0.3 and 0.7 ratio. Lower ratio = Higher score.
        ratio = row["billed_kwh"] / row["dt_avg_kwh"]
        return 0.1 + (0.9 - 0.1) * (0.7 - ratio) / (0.7 - 0.3)
    
    cust_sum["dt_relative_usage_score"] = cust_sum.apply(_score, axis=1)
    return cust_sum[["ACCOUNT_NUMBER", "dt_relative_usage_score"]]

def generate_escalations_report(ppm_df, ppd_df, escalations_df, customer_scores_df, months_list):
    escalations = escalations_df.copy()
    acct_col = None
    for col in escalations.columns:
        if col.strip().lower() in ["account no", "account_no", "accountnumber", "account number"]:
            acct_col = col
            break
    if acct_col is None:
        st.error("Escalations sheet does not contain 'Account No' column.")
        return pd.DataFrame()
    accounts = escalations[acct_col].astype(str).str.strip().unique().tolist()
    customers = pd.concat([ppm_df, ppd_df], ignore_index=True, sort=False)
    if "ACCOUNT_NUMBER" not in customers.columns:
        for c in customers.columns:
            if c.strip().lower() in ["account no", "account_no", "accountnumber", "account number", "acct"]:
                customers = customers.rename(columns={c: "ACCOUNT_NUMBER"})
                break
    if "ACCOUNT_NUMBER" not in customers.columns:
        customers["ACCOUNT_NUMBER"] = ""
    
    reports = []
    for acc in accounts:
        matched = customers[customers["ACCOUNT_NUMBER"].astype(str).str.strip() == str(acc).strip()]
        if matched.empty:
            reports.append({
                "Account No": acc,
                "Found": "No",
                "Billing_Type": "",
                "ACCOUNT_NUMBER": acc,
                "CUSTOMER_NAME": "Not Found",
                "Feeder": "",
                "NAME_OF_DT": "",
                "METER_NUMBER": "",
                **{m: np.nan for m in months_list},
                "Theft Probability (avg)": np.nan
            })
        else:
            for _, r in matched.iterrows():
                row = {
                    "Account No": acc,
                    "Found": "Yes",
                    "Billing_Type": r.get("Billing_Type", ""),
                    "ACCOUNT_NUMBER": r.get("ACCOUNT_NUMBER", ""),
                    "CUSTOMER_NAME": r.get("CUSTOMER_NAME", ""),
                    "Feeder": r.get("Feeder", ""),
                    "NAME_OF_DT": r.get("NAME_OF_DT", ""),
                    "METER_NUMBER": r.get("METER_NUMBER", "")
                }
                for m in months_list:
                    colname = f"{m} (kWh)"
                    row[m] = r.get(colname, np.nan) if colname in r.index else (r.get(m, np.nan) if m in r.index else np.nan)
                
                tp = np.nan
                try:
                    tp_row = customer_scores_df[customer_scores_df["ACCOUNT_NUMBER"].astype(str) == str(acc)]
                    # Check for both weighted and ML scores
                    if not tp_row.empty and "theft_probability_avg" in tp_row.columns:
                         tp = float(tp_row["theft_probability_avg"].mean())
                    elif not tp_row.empty and "theft_probability_ml_avg" in tp_row.columns:
                        tp = float(tp_row["theft_probability_ml_avg"].mean())
                except Exception:
                    tp = np.nan
                
                row["Theft Probability (avg)"] = tp
                reports.append(row)
    return pd.DataFrame(reports)

# --- NEW ML Function ---
@st.cache_data
def run_isolation_forest(df, features, contamination_rate=0.01):
    """Applies Isolation Forest to the specified features and returns the anomaly score."""
    st.info(f"Running Isolation Forest on {len(df)} customers with a contamination rate of {contamination_rate*100}%.")
    
    # 1. Prepare data and handle NaNs
    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

    # 2. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train Isolation Forest
    try:
        model = IsolationForest(
            n_estimators=100, 
            contamination=contamination_rate, 
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_scaled)
    except Exception as e:
        st.error(f"Isolation Forest failed to fit: {e}")
        df['anomaly_score'] = 0.5
        df['theft_probability_ml'] = 0.0
        return df

    # 4. Get raw anomaly score (lower score = more normal)
    anomaly_score = model.decision_function(X_scaled)
    
    # 5. Convert raw anomaly score to a normalized theft probability (0 to 1)
    # iForest anomaly_score ranges roughly from -0.5 to -0.2 (normal) to very low (anomaly)
    # We use min-max scaling to stretch the score into a 0-1 probability,
    # where 1 is the highest risk (most negative anomaly score).
    min_score = anomaly_score.min()
    max_score = anomaly_score.max()
    
    # Normalize: (score - min) / (max - min). Anomaly scores are more negative.
    # We want highly negative scores (anomalies) to be close to 1.
    # Therefore, we invert the normalized score: 1 - ((score - min) / (max - min))
    normalized_score = (anomaly_score - min_score) / (max_score - min_score)
    df['theft_probability_ml'] = 1 - normalized_score

    st.success("Isolation Forest analysis complete.")
    return df

# --- Begin main app logic ---
st.title("SniffIt🐶")

st.subheader("Energy Theft Detector (ML Upgrade)")
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is None:
    st.warning("Please upload an Excel file to proceed.")
    st.stop()

# --- Data Loading and Sheet Checks (Kept as is) ---
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

def _get_sheet_case_insensitive(sheets_dict, target_name):
    for k in sheets_dict.keys():
        if k.strip().lower() == target_name.strip().lower():
            return sheets_dict[k]
    return None

feeder_df = sheets.get("Feeder Data")
if feeder_df is None:
    feeder_df = _get_sheet_case_insensitive(sheets, "Feeder Data")
dt_df = sheets.get("Transformer Data")
if dt_df is None:
    dt_df = _get_sheet_case_insensitive(sheets, "Transformer Data")
ppm_df = sheets.get("Customer Data_PPM")
if ppm_df is None:
    ppm_df = _get_sheet_case_insensitive(sheets, "Customer Data_PPM")
ppd_df = sheets.get("Customer Data_PPD")
if ppd_df is None:
    ppd_df = _get_sheet_case_insensitive(sheets, "Customer Data_PPD")
band_df = sheets.get("Feeder Band")
if band_df is None:
    band_df = _get_sheet_case_insensitive(sheets, "Feeder Band")
tariff_df = sheets.get("Customer Tariffs")
if tariff_df is None:
    tariff_df = _get_sheet_case_insensitive(sheets, "Customer Tariffs")
escalations_df = sheets.get("Escalations")
if escalations_df is None:
    escalations_df = _get_sheet_case_insensitive(sheets, "Escalations")

if any(df is None for df in [feeder_df, dt_df, ppm_df, ppd_df, band_df, tariff_df, escalations_df]):
    st.error("One or more required sheets missing. Check that your Excel file has these sheets: Feeder Data, Transformer Data, Customer Data_PPM, Customer Data_PPD, Feeder Band, Customer Tariffs, Escalations.")
    st.stop()

# --- Data Preprocessing and Merging (Simplified/Consolidated) ---

# Validate customer columns
required_customer_cols = ["NAME_OF_DT", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "NAME_OF_FEEDER", "BUSINESS_UNIT", "UNDERTAKING", "TARIFF"]
for col in required_customer_cols:
    if col not in ppm_df.columns:
        ppm_df[col] = ""
    if col not in ppd_df.columns:
        ppd_df[col] = ""

# Set default rate
default_rate = 209.5
if "Rate (NGN)" not in tariff_df.columns:
    rate_col = next((c for c in tariff_df.columns if "rate" in str(c).lower()), None)
    if rate_col:
        tariff_df["Rate (NGN)"] = pd.to_numeric(tariff_df[rate_col], errors="coerce").fillna(default_rate)
    else:
        tariff_df["Rate (NGN)"] = default_rate

# Normalize month columns
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN"]
for df, unit in [(feeder_df, 1000), (ppm_df, 1), (ppd_df, 1)]:
    for m in months:
        col = f"{m} (kWh)"
        if m in df.columns:
            df[col] = pd.to_numeric(df[m], errors="coerce").fillna(0) * unit
        else:
            if col not in df.columns:
                df[col] = 0
for m in months:
    col = f"{m} (kWh)"
    if m in dt_df.columns:
        dt_df[col] = pd.to_numeric(dt_df[m], errors="coerce").fillna(0)
    else:
        if col not in dt_df.columns:
            dt_df[col] = 0
for df in [feeder_df, dt_df, ppm_df, ppd_df]:
    drop_cols = [c for c in df.columns if c in months]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Normalize names
name_normalizations = [
    ("Feeder", feeder_df), ("NAME_OF_FEEDER", ppm_df), ("NAME_OF_FEEDER", ppd_df), ("Feeder", band_df),
    ("NAME_OF_DT", ppm_df), ("NAME_OF_DT", ppd_df), ("New Unique DT Nomenclature", dt_df),
    ("TARIFF", ppm_df), ("TARIFF", ppd_df), ("Tariff", tariff_df)
]
for col, df in name_normalizations:
    if col in df.columns:
        df[col] = df[col].apply(normalize_name)

# Combine customers
ppm_df["Billing_Type"] = "PPM"
ppd_df["Billing_Type"] = "PPD"
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True, sort=False)
customer_df = add_feeder_column_safe(customer_df, "NAME_OF_DT")

# Filter to valid feeders
dt_df["NAME_OF_DT"] = dt_df.get("New Unique DT Nomenclature", dt_df.get("NAME_OF_DT", ""))
dt_df["Feeder"] = dt_df["NAME_OF_DT"].apply(lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x and len(x.split("-")) >= 3 else x)
dt_df["Feeder"] = dt_df["Feeder"].apply(normalize_name)
dt_df["total_energy_kwh"] = dt_df[[f"{m} (kWh)" for m in months]].sum(axis=1)

# DT short names
dt_df["DT_Short_Name"] = dt_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))
customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))

# Melt customer monthly readings
required_id_vars = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type", "Feeder", "BUSINESS_UNIT", "UNDERTAKING", "TARIFF"]
for col in required_id_vars:
    if col not in customer_df.columns:
        customer_df[col] = ""
value_vars = [f"{m} (kWh)" for m in months]

try:
    customer_monthly = customer_df.melt(id_vars=required_id_vars, value_vars=value_vars, var_name="month", value_name="billed_kwh")
    customer_monthly["month"] = customer_monthly["month"].str.replace(" (kWh)", "")
    customer_monthly["month"] = pd.Categorical(customer_monthly["month"], categories=months, ordered=True)
except Exception as e:
    st.error(f"Failed to melt customer monthly data: {e}")
    st.stop()

# DT monthly melt
try:
    dt_agg_monthly = dt_df.melt(id_vars=["NAME_OF_DT", "DT_Short_Name", "Feeder", "Ownership", "Connection Status", "total_energy_kwh"], value_vars=value_vars, var_name="month", value_name="total_dt_kwh")
    dt_agg_monthly["month"] = dt_agg_monthly["month"].str.replace(" (kWh)", "")
    dt_agg_monthly["month"] = pd.Categorical(dt_agg_monthly["month"], categories=months, ordered=True)
except Exception as e:
    st.error(f"Failed to melt DT monthly: {e}")
    st.stop()

# --- UI Filters and Weights ---
st.subheader("Filters")
col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 3, 3, 1, 1])
# ... (Filter logic for BU, UT, Feeder, DT, Start/End Month remains the same)
with col1:
    bu_options = sorted(customer_df["BUSINESS_UNIT"].unique()) if "BUSINESS_UNIT" in customer_df.columns else []
    selected_bu = st.selectbox("Select Business Unit", bu_options, index=0 if bu_options else None)
with col2:
    if selected_bu:
        customer_df_bu = customer_df[customer_df["BUSINESS_UNIT"] == selected_bu]
        ut_options = sorted(customer_df_bu["UNDERTAKING"].unique()) if "UNDERTAKING" in customer_df_bu.columns else []
        selected_ut = st.selectbox("Select Undertaking", ut_options, index=0 if ut_options else None)
    else:
        selected_ut = ""
        ut_options = []
        st.selectbox("Select Undertaking", ["No Business Unit Selected"], index=0, disabled=True)
with col3:
    feeder_options = sorted(feeder_df["Feeder"].unique())
    selected_feeder_short = st.selectbox("Select Feeder (Full Name)", feeder_options, index=0 if feeder_options else None)
with col4:
    selected_feeder = selected_feeder_short
    selected_dt_short = ""
    if selected_feeder:
        dt_df_filtered = dt_df[dt_df["Feeder"] == selected_feeder]
        dt_options = sorted(dt_df_filtered["DT_Short_Name"].unique())
        selected_dt_short = st.selectbox("Select DT", dt_options, index=0 if dt_options else None)
    else:
        st.selectbox("Select DT", ["No Feeder Selected"], index=0, disabled=True)
with col5:
    start_month = st.selectbox("Start Month", months, index=0)
with col6:
    end_month = st.selectbox("End Month", months, index=len(months)-1)
    if months.index(start_month) > months.index(end_month):
        st.error("Start Month must be before or equal to End Month.")
        st.stop()
        
# --- New ML Control and Weights ---

st.subheader("Model Selection")
model_choice = st.radio(
    "Choose Risk Scoring Method",
    ('Weighted Rule-Based Model', 'Isolation Forest ML Model'),
    help="Weighted model uses pre-defined operational rules; Isolation Forest uses statistical anomaly detection on feature space."
)

if model_choice == 'Weighted Rule-Based Model':
    st.subheader("Adjust Weighted Score Factors")
    colw1, colw2, colw3, colw4, colw5, colw6 = st.columns(6)
    with colw1:
        w_feeder = st.slider("Feeder Efficiency Weight", 0.0, 1.0, 0.2, 0.01)
    with colw2:
        w_dt = st.slider("DT Efficiency Weight", 0.0, 1.0, 0.2, 0.01)
    with colw3:
        w_location = st.slider("Location Trust Weight", 0.0, 1.0, 0.4, 0.01)
    with colw4:
        w_pattern = st.slider("Consumption Pattern Weight", 0.0, 1.0, 0.7, 0.01)
    with colw5:
        w_relative = st.slider("DT Relative Usage Weight", 0.0, 1.0, 0.7, 0.01)
    with colw6:
        w_zero = st.slider("Zero Frequency Weight", 0.0, 1.0, 0.7, 0.01)
    
    total_weight = w_feeder + w_dt + w_location + w_pattern + w_relative + w_zero
    if total_weight == 0:
        st.error("Total weight cannot be zero. Adjust sliders.")
        st.stop()
    w_feeder /= total_weight
    w_dt /= total_weight
    w_location /= total_weight
    w_pattern /= total_weight
    w_relative /= total_weight
    w_zero /= total_weight
else: # Isolation Forest ML Model
    st.subheader("Isolation Forest Parameters")
    contamination_rate = st.slider(
        "Contamination Rate (Estimated % of Thieves)",
        0.005, 0.10, 0.01, 0.001,
        help="This is the expected fraction of anomalies in the dataset. Affects the model's threshold for labeling an anomaly."
    )
    # Set dummy weights (used only for the structure of the next steps)
    w_feeder = w_dt = w_location = w_pattern = w_relative = w_zero = 1.0
    
# --- Calculation Pipeline ---

# Compute location trust scores (based on escalations)
escalations_df_local = escalations_df.copy()
escalations_df_local["Report_Count"] = 1
feeder_escal = escalations_df_local.groupby("Feeder", as_index=False)["Report_Count"].sum()
if not feeder_escal.empty:
    feeder_escal["location_trust_score"] = feeder_escal["Report_Count"] / feeder_escal["Report_Count"].max()
else:
    feeder_escal = pd.DataFrame({"Feeder": feeder_df["Feeder"], "location_trust_score": 0.0})
dt_escal = escalations_df_local.groupby("DT Nomenclature", as_index=False)["Report_Count"].sum()
if not dt_escal.empty:
    dt_escal["location_trust_score"] = dt_escal["Report_Count"] / dt_escal["Report_Count"].max()
else:
    dt_escal = pd.DataFrame({"DT Nomenclature": dt_df["NAME_OF_DT"], "location_trust_score": 0.0})

# Compute customer features
pattern_df_full = calculate_pattern_deviation(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
zero_df_full = calculate_zero_counter(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
selected_months = months[months.index(start_month):months.index(end_month)+1]
customer_monthly_sel = customer_monthly[customer_monthly["month"].isin(selected_months)].copy()
dt_relative_df_sel = calculate_dt_relative_usage(customer_monthly_sel)

# Merge features to monthly data (for plotting/filtering)
customer_monthly_sel = customer_monthly_sel.merge(pattern_df_full, on="ACCOUNT_NUMBER", how="left")
customer_monthly_sel = customer_monthly_sel.merge(zero_df_full, on="ACCOUNT_NUMBER", how="left")
customer_monthly_sel = customer_monthly_sel.merge(dt_relative_df_sel, on="ACCOUNT_NUMBER", how="left")

# --- Aggregate and Calculate Billing Efficiencies (Same as original) ---
customer_billed_monthly = customer_monthly_sel.groupby(["NAME_OF_DT", "DT_Short_Name", "month"], as_index=False)["billed_kwh"].sum().rename(columns={"billed_kwh":"customer_billed_kwh"})
dt_merged_monthly = dt_agg_monthly.merge(customer_billed_monthly, left_on=["NAME_OF_DT", "DT_Short_Name", "month"], right_on=["NAME_OF_DT", "DT_Short_Name", "month"], how="left")
dt_merged_monthly["customer_billed_kwh"] = dt_merged_monthly["customer_billed_kwh"].fillna(0)
dt_merged_monthly["total_billed_kwh"] = np.where(
    dt_merged_monthly.get("Ownership", "").str.strip().str.upper().isin(["PRIVATE"]),
    dt_merged_monthly["total_dt_kwh"],
    dt_merged_monthly["customer_billed_kwh"]
)
dt_merged_monthly["dt_billing_efficiency"] = np.where(
    (dt_merged_monthly.get("Connection Status", "").str.strip().str.upper()=="NOT CONNECTED") & (dt_merged_monthly["total_energy_kwh"]>0),
    0.0,
    (dt_merged_monthly["total_billed_kwh"] / dt_merged_monthly["total_dt_kwh"].replace(0,1)).clip(0,1)
)

# Feeder calculations
feeder_monthly = feeder_df.melt(id_vars=["Feeder"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="feeder_energy_kwh")
feeder_monthly["month"] = feeder_monthly["month"].str.replace(" (kWh)", "")
feeder_monthly = feeder_monthly[feeder_monthly["month"].isin(selected_months)]
feeder_agg = feeder_monthly.groupby("Feeder", as_index=False)["feeder_energy_kwh"].sum()
dt_merged = dt_merged_monthly.groupby(["NAME_OF_DT", "DT_Short_Name", "Feeder", "Ownership", "Connection Status", "total_energy_kwh"], as_index=False)["total_dt_kwh"].sum()
cust_agg_total = customer_monthly_sel.groupby(["NAME_OF_DT", "DT_Short_Name", "Feeder"], as_index=False)["billed_kwh"].sum().rename(columns={"billed_kwh":"customer_billed_kwh"})
dt_merged = dt_merged.merge(cust_agg_total, left_on=["NAME_OF_DT", "DT_Short_Name", "Feeder"], right_on=["NAME_OF_DT", "DT_Short_Name", "Feeder"], how="left")
dt_merged["customer_billed_kwh"] = dt_merged["customer_billed_kwh"].fillna(0)
dt_merged["total_billed_kwh"] = np.where(dt_merged.get("Ownership", "").str.strip().str.upper().isin(["PRIVATE"]), dt_merged["total_dt_kwh"], dt_merged["customer_billed_kwh"])
dt_merged["dt_billing_efficiency"] = np.where((dt_merged.get("Connection Status", "").str.strip().str.upper()=="NOT CONNECTED") & (dt_merged["total_energy_kwh"]>0), 0.0, (dt_merged["total_billed_kwh"] / dt_merged["total_dt_kwh"].replace(0,1)).clip(0,1))
feeder_agg_billed = dt_merged.groupby("Feeder", as_index=False)["total_billed_kwh"].sum()
feeder_merged = feeder_agg.merge(feeder_agg_billed, on="Feeder", how="left")
feeder_merged["total_billed_kwh"] = feeder_merged["total_billed_kwh"].fillna(0)
feeder_merged["feeder_billing_efficiency"] = (feeder_merged["total_billed_kwh"] / feeder_merged["feeder_energy_kwh"].replace(0,1)).clip(0,1)
feeder_merged["location_trust_score"] = feeder_merged.merge(feeder_escal[["Feeder", "location_trust_score"]], on="Feeder", how="left")["location_trust_score"].fillna(0.0)

# Merge DT efficiency and location scores back to customer level
customer_monthly_sel = customer_monthly_sel.merge(feeder_merged[["Feeder", "feeder_billing_efficiency", "location_trust_score"]], on="Feeder", how="left")
customer_monthly_sel = customer_monthly_sel.merge(dt_merged[["NAME_OF_DT", "DT_Short_Name", "dt_billing_efficiency"]], left_on=["NAME_OF_DT", "DT_Short_Name"], right_on=["NAME_OF_DT", "DT_Short_Name"], how="left")
merged_dt = customer_monthly_sel.merge(dt_escal[["DT Nomenclature", "location_trust_score"]].rename(columns={"location_trust_score": "location_trust_score_dt"}), left_on="NAME_OF_DT", right_on="DT Nomenclature", how="left")
customer_monthly_sel["location_trust_score_dt"] = merged_dt["location_trust_score_dt"].fillna(0.0)
customer_monthly_sel["location_trust_score_feeder"] = customer_monthly_sel["location_trust_score"].fillna(0.0)
customer_monthly_sel["location_trust_score"] = customer_monthly_sel["location_trust_score_dt"].combine_first(customer_monthly_sel["location_trust_score_feeder"]).fillna(0.0)

# Standardize feature names (for both weighted and ML models)
customer_monthly_sel.rename(columns={
    "pattern_deviation_score": "F_Pattern",
    "dt_relative_usage_score": "F_Relative",
    "zero_counter_score": "F_Zero",
    "feeder_billing_efficiency": "F_Feeder_Eff",
    "dt_billing_efficiency": "F_DT_Eff",
    "location_trust_score": "F_Location_Risk" # Note: F_Location_Risk is already a risk score (higher is bad)
}, inplace=True)

# 1. CALCULATE WEIGHTED SCORE (Rule-Based)
customer_monthly_sel["theft_probability"] = (
    w_feeder * (1 - customer_monthly_sel["F_Feeder_Eff"]) +
    w_dt * (1 - customer_monthly_sel["F_DT_Eff"]) +
    w_location * customer_monthly_sel["F_Location_Risk"] +
    w_pattern * customer_monthly_sel["F_Pattern"] +
    w_relative * customer_monthly_sel["F_Relative"] +
    w_zero * customer_monthly_sel["F_Zero"]
).clip(0,1)

# 2. CALCULATE ISOLATION FOREST SCORE (Unsupervised ML)
ml_features = ["F_Pattern", "F_Relative", "F_Zero", "F_Location_Risk", "F_Feeder_Eff", "F_DT_Eff"]
customer_features = customer_monthly_sel.groupby("ACCOUNT_NUMBER")[ml_features].mean().reset_index()

customer_features = run_isolation_forest(
    customer_features, 
    ml_features, 
    contamination_rate=contamination_rate if model_choice == 'Isolation Forest ML Model' else 0.01 # Use slider value if ML is selected
)

# Merge ML score back to the monthly data for filtering/plotting
customer_monthly_sel = customer_monthly_sel.merge(
    customer_features[["ACCOUNT_NUMBER", "theft_probability_ml"]], 
    on="ACCOUNT_NUMBER", 
    how="left"
)

# Select the score to use based on user choice
if model_choice == 'Weighted Rule-Based Model':
    score_column = "theft_probability"
    score_label = "Theft Probability (Weighted)"
else:
    score_column = "theft_probability_ml"
    score_label = "Theft Probability (ML)"

customer_monthly_sel["risk_tier"] = pd.cut(customer_monthly_sel[score_column], bins=[0,0.4,0.7,1.0], labels=["Low", "Medium", "High"], include_lowest=True)

# --- Aggregation for Display ---
month_customers = customer_monthly_sel.groupby(["ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type", "DT_Short_Name"], as_index=False).agg({
    "billed_kwh": "sum",
    score_column: "mean", # The chosen score
    "theft_probability": "mean", # Keep original weighted score for comparison
    "theft_probability_ml": "mean", # Keep ML score for comparison
    "F_Pattern": "mean",
    "F_Relative": "mean",
    "F_Zero": "mean"
})

# Rename the final score column for display
month_customers = month_customers.rename(columns={
    "billed_kwh": "billed_kwh_total", 
    score_column: "final_theft_probability_avg",
    "F_Pattern": "Pattern Deviation Score",
    "F_Relative": "DT Relative Usage Score",
    "F_Zero": "Zero Frequency Score"
})

# --- Visualization and Display ---
# ... (Heatmaps and dataframes use the selected 'score_column' for plotting)

# DT Theft Heatmap
st.subheader(f"DT Risk Heatmap (Based on {score_label})")
if selected_feeder:
    try:
        dt_filtered = dt_merged_monthly[dt_merged_monthly["Feeder"] == selected_feeder]
        if dt_filtered.empty:
            st.warning("No DT monthly data for this feeder.")
        else:
            dt_scores = dt_filtered.groupby("DT_Short_Name")["dt_billing_efficiency"].mean().reset_index()
            dt_scores["theft_probability"] = 1 - dt_scores["dt_billing_efficiency"]
            order = dt_scores.sort_values("theft_probability", ascending=False)["DT_Short_Name"].tolist()
            # DT heatmap remains based on billing efficiency for stability
            dt_pivot = dt_filtered.pivot_table(index="DT_Short_Name", columns="month", values="dt_billing_efficiency", aggfunc="mean").reindex(index=order, columns=selected_months)
            if not dt_pivot.empty:
                plt.figure(figsize=(10, max(4, len(dt_pivot)/2)))
                sns.heatmap(1 - dt_pivot, vmin=0, vmax=1, cmap="Reds", cbar_kws={"label": "DT Billing Efficiency Risk"})
                plt.title(f"DT Billing Efficiency Risk for {selected_feeder} ({start_month} to {end_month})")
                st.pyplot(plt.gcf())
                plt.close()
            else:
                st.warning("No DT pivot data.")
    except Exception as e:
        st.error(f"DT heatmap failed: {e}")

# Customer Heatmap
st.subheader(f"Customer Theft Probability Heatmap (Based on {score_label})")
if selected_dt_short:
    try:
        filtered_customers = customer_monthly_sel[customer_monthly_sel["DT_Short_Name"] == selected_dt_short]
        if filtered_customers.empty:
            st.warning("No customer monthly data for this DT.")
        else:
            customer_scores = filtered_customers.groupby("ACCOUNT_NUMBER")[score_column].mean().reset_index()
            customer_order = customer_scores.sort_values(score_column, ascending=False)["ACCOUNT_NUMBER"].tolist()
            
            num_customers = st.number_input("Number of high-risk customers for Heatmap (0 for all)", min_value=0, value=min(10, len(customer_order)), step=1)
            
            if num_customers > 0:
                chosen = customer_order[:num_customers]
                filtered_for_heatmap = filtered_customers[filtered_customers["ACCOUNT_NUMBER"].isin(chosen)]
            else:
                filtered_for_heatmap = filtered_customers
            
            # Pivot table uses the selected score_column
            pivot_data = filtered_for_heatmap.pivot_table(index="ACCOUNT_NUMBER", columns="month", values=score_column, aggfunc="mean").reindex(index=customer_order[:num_customers or None], columns=selected_months)
            
            if not pivot_data.empty:
                plt.figure(figsize=(10, max(4, len(pivot_data)/2)))
                sns.heatmap(pivot_data, vmin=0, vmax=1, cmap="Reds", cbar_kws={"label": score_label})
                plt.title(f"Theft Probability for {selected_dt_short} ({selected_feeder})")
                st.pyplot(plt.gcf())
                plt.close()
            else:
                st.warning("No data to render heatmap.")
    except Exception as e:
        st.error(f"Customer heatmap failed: {e}")

# Display customer list
st.subheader(f"Customer Risk List ({score_label})")
if selected_dt_short:
    filtered_customers = month_customers[month_customers["DT_Short_Name"] == selected_dt_short]
    if filtered_customers.empty:
        st.warning("No customers for this DT in selected period.")
    else:
        display_df = filtered_customers.sort_values("final_theft_probability_avg", ascending=False)
        
        # Select columns to display based on chosen model (show both final scores for comparison)
        cols_to_display = ["ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "Billing_Type", 
                           "final_theft_probability_avg", "theft_probability", "theft_probability_ml", 
                           "Pattern Deviation Score", "Zero Frequency Score", "DT Relative Usage Score", "billed_kwh_total"]
        
        # Rename the final score column back to the generic name for consistent display
        display_df = display_df.rename(columns={"final_theft_probability_avg": score_label})
        
        st.dataframe(display_df[cols_to_display].style.format({
            "billed_kwh_total": "{:.2f}",
            score_label: "{:.3f}",
            "theft_probability": "{:.3f}",
            "theft_probability_ml": "{:.3f}",
            "Pattern Deviation Score": "{:.3f}",
            "DT Relative Usage Score": "{:.3f}",
            "Zero Frequency Score": "{:.3f}"
        }), use_container_width=True)

# ... (Feeder Summary, DT Summary, and Export sections remain largely the same, using the feeder/dt merged data which isn't affected by the customer-level score choice)

# Export customer list CSV
st.subheader("Export Customer Data")
try:
    if selected_dt_short:
        export_customers = month_customers[month_customers["DT_Short_Name"] == selected_dt_short]
        if not export_customers.empty:
            csv = export_customers.to_csv(index=False)
            st.download_button(
                label=f"Download Customer List for {selected_dt_short} ({start_month} to {end_month})",
                data=csv,
                file_name=f"theft_analysis_{selected_dt_short}_{start_month}_to_{end_month}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"No customer data to export for DT {selected_dt_short}.")
    else:
        st.info("Select a DT to export customer data.")
except Exception as e:
    st.error(f"CSV export failed: {e}")

# Escalations report (Updated to use the multi-score output)
st.subheader("Escalations Report (full lookup of 'Account No')")
try:
    cust_scores_avg = customer_monthly_sel.groupby("ACCOUNT_NUMBER", as_index=False).agg(
        theft_probability_avg=('theft_probability', 'mean'),
        theft_probability_ml_avg=('theft_probability_ml', 'mean')
    ).reset_index(drop=True)
    
    escal_report_df = generate_escalations_report(ppm_df, ppd_df, escalations_df, cust_scores_avg, months)
    
    # Rename the generic Theft Probability to show which model was active in the UI
    escal_report_df = escal_report_df.rename(columns={"Theft Probability (avg)": f"Theft Probability (Based on {model_choice})"})
    
    if escal_report_df.empty:
        st.info("Escalations report produced no rows.")
    else:
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
            escal_report_df.to_excel(writer, index=False, sheet_name="Escalations Report")
        st.download_button(
            label="📥 Download Escalations Report (Excel)",
            data=towrite.getvalue(),
            file_name="Escalations_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
except Exception as e:
    st.error(f"Failed to generate escalations report: {e}")

# Quick Performance Metrics (Kept as is)
# ... (This section remains unchanged as it calculates total billed/lost and efficiency)

# Footer
st.markdown("Built by Elvis Ebenuwah for Ikeja Electric. SniffIt🐶 2025.")
