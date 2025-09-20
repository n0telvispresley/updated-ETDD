import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import re
import io

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="IE Energy Theft Detection Dashboard", layout="wide")

# -------------------------
# Utility functions
# -------------------------
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
    return pd.DataFrame(results)

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
    return pd.DataFrame(results)

def calculate_dt_relative_usage(customer_monthly):
    cust_sum = customer_monthly.groupby(["ACCOUNT_NUMBER", "NAME_OF_DT"], as_index=False)["billed_kwh"].sum()
    dt_avg = cust_sum[cust_sum["billed_kwh"] > 0].groupby("NAME_OF_DT", as_index=False)["billed_kwh"].mean().rename(columns={"billed_kwh": "dt_avg_kwh"})
    cust_sum = cust_sum.merge(dt_avg, on="NAME_OF_DT", how="left")
    cust_sum["relative_ratio"] = np.where(cust_sum["dt_avg_kwh"].fillna(0) == 0, 0.5, cust_sum["billed_kwh"] / cust_sum["dt_avg_kwh"])
    def _score(row):
        if pd.isna(row["dt_avg_kwh"]) or row["dt_avg_kwh"] == 0:
            return 0.5 if row["billed_kwh"] == 0 else 0.1
        if row["billed_kwh"] < 0.3 * row["dt_avg_kwh"]:
            return 0.9
        if row["billed_kwh"] > 0.7 * row["dt_avg_kwh"]:
            return 0.1
        ratio = row["billed_kwh"] / row["dt_avg_kwh"]
        return 0.1 + (0.9 - 0.1) * (0.7 - ratio) / (0.7 - 0.3)
    cust_sum["dt_relative_usage_score"] = cust_sum.apply(_score, axis=1)
    return cust_sum[["ACCOUNT_NUMBER", "dt_relative_usage_score"]]

def optimize_customer_weights(customer_df_for_opt, escalations_df, w_feeder, w_dt, w_location, step=0.05):
    accounts = set(escalations_df["Account No"].astype(str))
    sub = customer_df_for_opt[customer_df_for_opt["ACCOUNT_NUMBER"].astype(str).isin(accounts)].copy()
    if sub.empty:
        return (st.session_state.get("w_pattern", 0.2),
                st.session_state.get("w_relative", 0.2),
                st.session_state.get("w_zero", 0.05),
                None, None)
    best_mean = -np.inf
    best_combo = (st.session_state.get("w_pattern", 0.2),
                  st.session_state.get("w_relative", 0.2),
                  st.session_state.get("w_zero", 0.05))
    pre_mean = (
        w_feeder * (1 - sub["feeder_billing_efficiency"]) +
        w_dt * (1 - sub["dt_billing_efficiency"]) +
        w_location * sub["location_trust_score"] +
        st.session_state.get("w_pattern", 0.2) * sub["pattern_deviation_score"] +
        st.session_state.get("w_relative", 0.2) * sub["dt_relative_usage_score"] +
        st.session_state.get("w_zero", 0.05) * sub["zero_counter_score"]
    ).mean()
    rng = np.arange(0.0, 1.0001, step)
    for wp in rng:
        for wr in rng:
            for wz in rng:
                total = wp + wr + wz
                if total == 0:
                    continue
                wp_n, wr_n, wz_n = wp/total, wr/total, wz/total
                theft_scores = (
                    w_feeder * (1 - sub["feeder_billing_efficiency"]) +
                    w_dt * (1 - sub["dt_billing_efficiency"]) +
                    w_location * sub["location_trust_score"] +
                    wp_n * sub["pattern_deviation_score"] +
                    wr_n * sub["dt_relative_usage_score"] +
                    wz_n * sub["zero_counter_score"]
                )
                mean_score = theft_scores.mean()
                if mean_score > best_mean:
                    best_mean = mean_score
                    best_combo = (wp_n, wr_n, wz_n)
    return best_combo[0], best_combo[1], best_combo[2], pre_mean, best_mean

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
                    if not tp_row.empty and "theft_probability" in tp_row.columns:
                        tp = float(tp_row["theft_probability"].mean())
                except Exception:
                    tp = np.nan
                row["Theft Probability (avg)"] = tp
                reports.append(row)
    return pd.DataFrame(reports)

# -----------------------------
# Begin main app logic
# -----------------------------
st.title("Ikeja Electric Energy Theft Detection Dashboard")

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

# Validate columns presence (best-effort)
required_customer_cols = ["NAME_OF_DT", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "NAME_OF_FEEDER", "BUSINESS_UNIT", "UNDERTAKING", "TARIFF"]
for col in required_customer_cols:
    if col not in ppm_df.columns:
        ppm_df[col] = ""
    if col not in ppd_df.columns:
        ppd_df[col] = ""

if "New Unique DT Nomenclature" not in dt_df.columns:
    dt_df["New Unique DT Nomenclature"] = dt_df.get("NAME_OF_DT", "")

if "Feeder" not in feeder_df.columns:
    feeder_df["Feeder"] = feeder_df.columns[0] if feeder_df.shape[1] > 0 else ""
feeder_df["Feeder"] = feeder_df["Feeder"].apply(preserve_exact_string)

if "Tariff" not in tariff_df.columns:
    tariff_df["Tariff"] = ""
if "Rate (NGN)" not in tariff_df.columns:
    rate_col = next((c for c in tariff_df.columns if "rate" in str(c).lower()), None)
    if rate_col:
        tariff_df["Rate (NGN)"] = pd.to_numeric(tariff_df[rate_col], errors="coerce").fillna(209.5)
    else:
        tariff_df["Rate (NGN)"] = 209.5

# Normalize month columns and convert to (kWh) columns
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

# Drop raw month columns if present
for df in [feeder_df, dt_df, ppm_df, ppd_df]:
    drop_cols = [c for c in df.columns if c in months]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Normalize names across sheets
name_normalizations = [
    ("Feeder", feeder_df), ("NAME_OF_FEEDER", ppm_df), ("NAME_OF_FEEDER", ppd_df), ("Feeder", band_df),
    ("NAME_OF_DT", ppm_df), ("NAME_OF_DT", ppd_df), ("New Unique DT Nomenclature", dt_df),
    ("TARIFF", ppm_df), ("TARIFF", ppd_df), ("Tariff", tariff_df),
    ("BUSINESS_UNIT", ppm_df), ("BUSINESS_UNIT", ppd_df),
    ("UNDERTAKING", ppm_df), ("UNDERTAKING", ppd_df),
    ("Feeder", escalations_df), ("DT Nomenclature", escalations_df)
]
for col, df in name_normalizations:
    if col in df.columns:
        df[col] = df[col].apply(normalize_name)

# Compute band rates and map feeders -> tariff rates (MOVED EARLIER)
if "Feeder" not in band_df.columns:
    band_df["Feeder"] = ""
if "BAND" not in band_df.columns:
    band_df["BAND"] = ""
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
feeder_df = feeder_df.merge(band_df[["Feeder", "BAND"]], on="Feeder", how="left")
feeder_df["BAND"] = feeder_df["BAND"].fillna("Unknown")
feeder_df["Tariff_Rate"] = feeder_df["BAND"].map(band_rates).fillna(209.5)
if "Short Name" in band_df.columns:
    feeder_df = feeder_df.merge(band_df[["Feeder", "Short Name"]], on="Feeder", how="left")
    feeder_df["Feeder_Short"] = feeder_df["Short Name"].fillna(feeder_df["Feeder"])
    feeder_df.drop(columns=["Short Name"], inplace=True, errors="ignore")
else:
    feeder_df["Feeder_Short"] = feeder_df["Feeder"]

# Combine customers
ppm_df["Billing_Type"] = "PPM"
ppd_df["Billing_Type"] = "PPD"
customer_df = pd.concat([ppm_df, ppd_df], ignore_index=True, sort=False)

# Create Feeder columns safely
customer_df = add_feeder_column_safe(customer_df, "NAME_OF_DT")
dt_df["NAME_OF_DT"] = dt_df.get("New Unique DT Nomenclature", dt_df.get("NAME_OF_DT", ""))
dt_df["Feeder"] = dt_df["NAME_OF_DT"].apply(lambda x: "-".join(x.split("-")[:-1]) if isinstance(x, str) and "-" in x and len(x.split("-")) >= 3 else x)
dt_df["Feeder"] = dt_df["Feeder"].apply(normalize_name)

# Merge Tariff_Rate into dt_df (NOW SAFE since Tariff_Rate exists in feeder_df)
dt_df = dt_df.merge(feeder_df[["Feeder", "Tariff_Rate"]], on="Feeder", how="left")
dt_df["Tariff_Rate"] = dt_df["Tariff_Rate"].fillna(209.5)

# Filter dt_df and customer_df to only feeders present in feeder_df
valid_feeders = set(feeder_df["Feeder"])
if "Feeder" in dt_df.columns:
    dt_df = dt_df[dt_df["Feeder"].isin(valid_feeders)].copy()
else:
    dt_df["Feeder"] = ""
if "Feeder" in customer_df.columns:
    customer_df = customer_df[customer_df["Feeder"].isin(valid_feeders)].copy()
else:
    customer_df["Feeder"] = ""

# Merge tariff rates into customer_df
if "TARIFF" in customer_df.columns:
    customer_df = customer_df.merge(tariff_df[["Tariff", "Rate (NGN)"]], left_on="TARIFF", right_on="Tariff", how="left")
    customer_df["Rate (NGN)"] = customer_df["Rate (NGN)"].fillna(209.5)
    customer_df.drop(columns=["Tariff"], inplace=True, errors="ignore")
else:
    customer_df["Rate (NGN)"] = 209.5

# Exclude NOT CONNECTED DTs with zero energy
dt_df["total_energy_kwh"] = dt_df[[f"{m} (kWh)" for m in months]].sum(axis=1)
not_connected_zero = (dt_df.get("Connection Status", "").str.strip().str.upper() == "NOT CONNECTED") & (dt_df["total_energy_kwh"] == 0)
if not_connected_zero.any():
    st.warning(f"Excluding {not_connected_zero.sum()} DTs marked NOT CONNECTED with zero energy.")
dt_df = dt_df[~not_connected_zero].copy()

# Create DT short names
dt_df["DT_Short_Name"] = dt_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))
customer_df["DT_Short_Name"] = customer_df["NAME_OF_DT"].apply(lambda x: get_short_name(x, is_dt=True))

# Melt customer monthly readings
required_id_vars = ["NAME_OF_DT", "DT_Short_Name", "ACCOUNT_NUMBER", "METER_NUMBER", "CUSTOMER_NAME", "ADDRESS", "Billing_Type", "Feeder", "Rate (NGN)"]
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
    dt_agg_monthly = dt_df.melt(id_vars=["NAME_OF_DT", "DT_Short_Name", "Feeder", "Tariff_Rate", "Ownership", "Connection Status", "total_energy_kwh"], value_vars=value_vars, var_name="month", value_name="total_dt_kwh")
    dt_agg_monthly["month"] = dt_agg_monthly["month"].str.replace(" (kWh)", "")
    dt_agg_monthly["month"] = pd.Categorical(dt_agg_monthly["month"], categories=months, ordered=True)
except Exception as e:
    st.error(f"Failed to melt DT monthly: {e}")
    st.stop()

# Filter month range UI
st.subheader("Filters")
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    bu_options = sorted(customer_df["BUSINESS_UNIT"].unique()) if "BUSINESS_UNIT" in customer_df.columns else []
    selected_bu = st.selectbox("Select Business Unit", [""] + bu_options) if bu_options else ""
with col2:
    if selected_bu:
        customer_df_bu = customer_df[customer_df["BUSINESS_UNIT"] == selected_bu]
        ut_options = sorted(customer_df_bu["UNDERTAKING"].unique()) if "UNDERTAKING" in customer_df_bu.columns else []
        selected_ut = st.selectbox("Select Undertaking", [""] + ut_options) if ut_options else ""
    else:
        selected_ut = ""
with col3:
    feeder_options = sorted(feeder_df["Feeder_Short"].unique()) if "Feeder_Short" in feeder_df.columns else sorted(feeder_df["Feeder"].unique())
    selected_feeder_short = st.selectbox("Select Feeder", [""] + feeder_options) if feeder_options else ""
with col4:
    selected_feeder = ""
    selected_dt_short = ""
    if selected_feeder_short:
        if "Feeder_Short" in feeder_df.columns:
            matching_feeders = feeder_df[feeder_df["Feeder_Short"] == selected_feeder_short]["Feeder"]
            selected_feeder = matching_feeders.iloc[0] if not matching_feeders.empty else selected_feeder_short
        else:
            selected_feeder = selected_feeder_short
        dt_df_filtered = dt_df[dt_df["Feeder"] == selected_feeder]
        dt_options = sorted(dt_df_filtered["DT_Short_Name"].unique())
        selected_dt_short = st.selectbox("Select DT", [""] + dt_options) if dt_options else ""
with col5:
    start_month = st.selectbox("Start Month", months)
with col6:
    end_month = st.selectbox("End Month", months, index=len(months)-1)
    if months.index(start_month) > months.index(end_month):
        st.error("Start Month must be before or equal to End Month.")
        st.stop()

# Weights sliders
st.subheader("Adjust Theft Probability Weights")
colw1, colw2, colw3 = st.columns(3)
if "w_pattern" not in st.session_state:
    st.session_state.w_pattern = 0.2
if "w_relative" not in st.session_state:
    st.session_state.w_relative = 0.2
if "w_zero" not in st.session_state:
    st.session_state.w_zero = 0.05
with colw1:
    w_feeder = st.slider("Feeder Billing Efficiency Weight", 0.0, 1.0, 0.2, 0.01)
    w_dt = st.slider("DT Billing Efficiency Weight", 0.0, 1.0, 0.2, 0.01)
with col2:
    w_location = st.slider("Location Trust Score Weight", 0.0, 1.0, 0.2, 0.01)
    w_pattern = st.slider("Consumption Pattern Deviation Weight", 0.0, 1.0, st.session_state.w_pattern, 0.01, key="w_pattern")
with colw3:
    w_relative = st.slider("DT Relative Usage Score Weight", 0.0, 1.0, st.session_state.w_relative, 0.01, key="w_relative")
    w_zero = st.slider("Zero Frequency Weight", 0.0, 1.0, st.session_state.w_zero, 0.01, key="w_zero")

# Normalize weights
total_weight = w_feeder + w_dt + w_location + w_pattern + w_relative + w_zero
if total_weight == 0:
    st.error("Total weight cannot be zero. Adjust sliders.")
    st.stop()
w_feeder = w_feeder / total_weight
w_dt = w_dt / total_weight
w_location = w_location / total_weight
w_pattern = w_pattern / total_weight
w_relative = w_relative / total_weight
w_zero = w_zero / total_weight

# Optimize weights
if st.button("Optimize Customer-Level Weights for Escalations"):
    try:
        month_indices = {m: i for i, m in enumerate(months)}
        selected_months = months[month_indices[start_month]:month_indices[end_month] + 1]
        if not selected_months:
            st.error("No months selected.")
            st.stop()
        customer_monthly_sel = customer_monthly[customer_monthly["month"].isin(selected_months)].copy()
        if customer_monthly_sel.empty:
            st.error("No customer monthly data for selected months.")
            st.stop()
        feeder_monthly = feeder_df.melt(id_vars=["Feeder", "Feeder_Short", "Tariff_Rate"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="feeder_energy_kwh")
        feeder_monthly["month"] = feeder_monthly["month"].str.replace(" (kWh)", "")
        feeder_monthly = feeder_monthly[feeder_monthly["month"].isin(selected_months)]
        feeder_agg = feeder_monthly.groupby(["Feeder"])["feeder_energy_kwh"].sum().reset_index()
        dt_sel = dt_agg_monthly[dt_agg_monthly["month"].isin(selected_months)].copy()
        dt_agg_sum = dt_sel.groupby(["NAME_OF_DT", "DT_Short_Name", "Feeder", "Tariff_Rate", "Ownership", "Connection Status", "total_energy_kwh"], as_index=False)["total_dt_kwh"].sum()
        cust_agg = customer_monthly_sel.groupby(["NAME_OF_DT", "Feeder", "ACCOUNT_NUMBER"], as_index=False)["billed_kwh"].sum()
        dt_customer_billed = cust_agg.groupby("NAME_OF_DT", as_index=False)["billed_kwh"].sum().rename(columns={"billed_kwh":"customer_billed_kwh"})
        dt_merged = dt_agg_sum.merge(dt_customer_billed, left_on="NAME_OF_DT", right_on="NAME_OF_DT", how="left")
        dt_merged["customer_billed_kwh"] = dt_merged["customer_billed_kwh"].fillna(0)
        dt_merged["total_billed_kwh"] = np.where(dt_merged["Ownership"].str.strip().str.upper().isin(["PRIVATE"]), dt_merged["total_dt_kwh"], dt_merged["customer_billed_kwh"])
        dt_merged["dt_billing_efficiency"] = np.where((dt_merged["Connection Status"].str.strip().str.upper()=="NOT CONNECTED") & (dt_merged["total_energy_kwh"]>0), 0.0, (dt_merged["total_billed_kwh"] / dt_merged["total_dt_kwh"].replace(0,1)).clip(0,1))
        feeder_billed = dt_merged.groupby("Feeder", as_index=False)["total_billed_kwh"].sum()
        feeder_merged = feeder_agg.merge(feeder_billed, on="Feeder", how="left")
        feeder_merged["total_billed_kwh"] = feeder_merged["total_billed_kwh"].fillna(0)
        feeder_merged["feeder_billing_efficiency"] = (feeder_merged["total_billed_kwh"] / feeder_merged["feeder_energy_kwh"].replace(0,1)).clip(0,1)
        feeder_merged["location_trust_score"] = 0
        escalations_df_local = escalations_df.copy()
        escalations_df_local["Report_Count"] = 1
        feeder_escal = escalations_df_local.groupby("Feeder", as_index=False)["Report_Count"].sum()
        if not feeder_escal["Report_Count"].empty:
            feeder_escal["location_trust_score"] = feeder_escal["Report_Count"] / feeder_escal["Report_Count"].max()
        feeder_merged = feeder_merged.merge(feeder_escal[["Feeder","location_trust_score"]], on="Feeder", how="left")
        feeder_merged["location_trust_score"] = feeder_merged["location_trust_score"].fillna(0)
        dt_escal = escalations_df_local.groupby("DT Nomenclature", as_index=False)["Report_Count"].sum()
        if not dt_escal["Report_Count"].empty:
            dt_escal["location_trust_score"] = dt_escal["Report_Count"] / dt_escal["Report_Count"].max()
        customer_full = customer_df.copy()
        pattern_df = calculate_pattern_deviation(customer_full, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
        zero_df = calculate_zero_counter(customer_full, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
        dt_relative_df = calculate_dt_relative_usage(customer_monthly_sel)
        cust_features = cust_agg.groupby("ACCOUNT_NUMBER", as_index=False)["billed_kwh"].sum().rename(columns={"billed_kwh":"customer_billed_kwh"})
        acct_to_dt = customer_df[["ACCOUNT_NUMBER","NAME_OF_DT","Feeder"]].drop_duplicates(subset=["ACCOUNT_NUMBER"])
        cust_features = cust_features.merge(acct_to_dt, on="ACCOUNT_NUMBER", how="left")
        cust_features = cust_features.merge(dt_merged[["NAME_OF_DT","dt_billing_efficiency"]], left_on="NAME_OF_DT", right_on="NAME_OF_DT", how="left")
        cust_features = cust_features.merge(feeder_merged[["Feeder","feeder_billing_efficiency","location_trust_score"]], on="Feeder", how="left")
        cust_features = cust_features.merge(pattern_df, left_on="ACCOUNT_NUMBER", right_on="id", how="left")
        cust_features = cust_features.merge(zero_df, left_on="ACCOUNT_NUMBER", right_on="id", how="left")
        cust_features = cust_features.merge(dt_relative_df, on="ACCOUNT_NUMBER", how="left")
        for c in ["feeder_billing_efficiency","dt_billing_efficiency","location_trust_score","pattern_deviation_score","zero_counter_score","dt_relative_usage_score"]:
            if c not in cust_features.columns:
                cust_features[c] = 0
        cust_features[["feeder_billing_efficiency","dt_billing_efficiency","location_trust_score","pattern_deviation_score","zero_counter_score","dt_relative_usage_score"]] = cust_features[["feeder_billing_efficiency","dt_billing_efficiency","location_trust_score","pattern_deviation_score","zero_counter_score","dt_relative_usage_score"]].fillna(0)
        try:
            wp_n, wr_n, wz_n, pre_mean, post_mean = optimize_customer_weights(cust_features, escalations_df_local, w_feeder, w_dt, w_location)
            st.session_state["w_pattern"] = float(wp_n)
            st.session_state["w_relative"] = float(wr_n)
            st.session_state["w_zero"] = float(wz_n)
            st.success(f"Optimizer applied pattern={wp_n:.3f}, relative={wr_n:.3f}, zero={wz_n:.3f}.")
            if pre_mean is not None and post_mean is not None:
                st.info(f"Escalation avg theft mean {pre_mean:.3f} -> {post_mean:.3f}")
        except Exception as e:
            st.error(f"Optimizer failed: {e}")
    except Exception as e:
        st.error(f"Optimizer pre-processing failed: {e}")

# After possible optimizer update, re-normalize weights
w_pattern = st.session_state.get("w_pattern", w_pattern)
w_relative = st.session_state.get("w_relative", w_relative)
w_zero = st.session_state.get("w_zero", w_zero)
total_weight = w_feeder + w_dt + w_location + w_pattern + w_relative + w_zero
if total_weight == 0:
    st.error("Total weight cannot be zero after optimization.")
    st.stop()
w_feeder = w_feeder / total_weight
w_dt = w_dt / total_weight
w_location = w_location / total_weight
w_pattern = w_pattern / total_weight
w_relative = w_relative / total_weight
w_zero = w_zero / total_weight

# Compute customer monthly / theft probability
pattern_df_full = calculate_pattern_deviation(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
zero_df_full = calculate_zero_counter(customer_df, "ACCOUNT_NUMBER", [f"{m} (kWh)" for m in months])
selected_months = months[months.index(start_month):months.index(end_month)+1]
customer_monthly_sel = customer_monthly[customer_monthly["month"].isin(selected_months)].copy()
dt_relative_df_sel = calculate_dt_relative_usage(customer_monthly_sel)
customer_monthly_sel = customer_monthly_sel.merge(pattern_df_full, left_on="ACCOUNT_NUMBER", right_on="id", how="left")
customer_monthly_sel = customer_monthly_sel.merge(zero_df_full, left_on="ACCOUNT_NUMBER", right_on="id", how="left")
customer_monthly_sel = customer_monthly_sel.merge(dt_relative_df_sel, on="ACCOUNT_NUMBER", how="left")
customer_billed_monthly = customer_monthly_sel.groupby(["NAME_OF_DT","month"], as_index=False)["billed_kwh"].sum().rename(columns={"billed_kwh":"customer_billed_kwh"})
dt_merged_monthly = dt_agg_monthly.merge(customer_billed_monthly, left_on=["NAME_OF_DT","month"], right_on=["NAME_OF_DT","month"], how="left")
dt_merged_monthly["customer_billed_kwh"] = dt_merged_monthly["customer_billed_kwh"].fillna(0)
dt_merged_monthly["total_billed_kwh"] = np.where(
    dt_merged_monthly.get("Ownership","").str.strip().str.upper().isin(["PRIVATE"]),
    dt_merged_monthly["total_dt_kwh"],
    dt_merged_monthly["customer_billed_kwh"]
)
dt_merged_monthly["dt_billing_efficiency"] = np.where(
    (dt_merged_monthly.get("Connection Status","").str.strip().str.upper()=="NOT CONNECTED") & (dt_merged_monthly["total_energy_kwh"]>0),
    0.0,
    (dt_merged_monthly["total_billed_kwh"] / dt_merged_monthly["total_dt_kwh"].replace(0,1)).clip(0,1)
)
feeder_monthly = feeder_df.melt(id_vars=["Feeder","Feeder_Short","Tariff_Rate"], value_vars=[f"{m} (kWh)" for m in months], var_name="month", value_name="feeder_energy_kwh")
feeder_monthly["month"] = feeder_monthly["month"].str.replace(" (kWh)","")
feeder_monthly = feeder_monthly[feeder_monthly["month"].isin(selected_months)]
feeder_agg = feeder_monthly.groupby(["Feeder","Feeder_Short","Tariff_Rate"], as_index=False)["feeder_energy_kwh"].sum()
dt_agg_sum = dt_merged_monthly.groupby(["NAME_OF_DT","DT_Short_Name","Feeder","Tariff_Rate","Ownership","Connection Status","total_energy_kwh"], as_index=False)["total_dt_kwh"].sum()
cust_agg_total = customer_monthly_sel.groupby(["NAME_OF_DT","Feeder"], as_index=False)["billed_kwh"].sum().rename(columns={"billed_kwh":"customer_billed_kwh"})
dt_merged = dt_agg_sum.merge(cust_agg_total, left_on=["NAME_OF_DT","Feeder"], right_on=["NAME_OF_DT","Feeder"], how="left")
dt_merged["customer_billed_kwh"] = dt_merged["customer_billed_kwh"].fillna(0)
dt_merged["total_billed_kwh"] = np.where(dt_merged.get("Ownership","").str.strip().str.upper().isin(["PRIVATE"]), dt_merged["total_dt_kwh"], dt_merged["customer_billed_kwh"])
dt_merged["dt_billing_efficiency"] = np.where((dt_merged.get("Connection Status","").str.strip().str.upper()=="NOT CONNECTED") & (dt_merged["total_energy_kwh"]>0), 0.0, (dt_merged["total_billed_kwh"] / dt_merged["total_dt_kwh"].replace(0,1)).clip(0,1))
feeder_agg_billed = dt_merged.groupby("Feeder", as_index=False)["total_billed_kwh"].sum()
feeder_merged = feeder_agg.merge(feeder_agg_billed, on="Feeder", how="left")
feeder_merged["total_billed_kwh"] = feeder_merged["total_billed_kwh"].fillna(0)
feeder_merged["feeder_billing_efficiency"] = (feeder_merged["total_billed_kwh"] / feeder_merged["feeder_energy_kwh"].replace(0,1)).clip(0,1)
customer_monthly_sel = customer_monthly_sel.merge(feeder_merged[["Feeder","feeder_billing_efficiency"]], on="Feeder", how="left")
customer_monthly_sel = customer_monthly_sel.merge(dt_merged[["NAME_OF_DT","dt_billing_efficiency"]], left_on="NAME_OF_DT", right_on="NAME_OF_DT", how="left")
escal_df_local = escalations_df.copy()
escal_df_local["Report_Count"] = 1
feeder_escal = escal_df_local.groupby("Feeder", as_index=False)["Report_Count"].sum()
if not feeder_escal.empty:
    feeder_escal["location_trust_score"] = feeder_escal["Report_Count"] / feeder_escal["Report_Count"].max()
else:
    feeder_escal = pd.DataFrame(columns=["Feeder","location_trust_score"])
dt_escal = escal_df_local.groupby("DT Nomenclature", as_index=False)["Report_Count"].sum()
if not dt_escal.empty:
    dt_escal["location_trust_score"] = dt_escal["Report_Count"] / dt_escal["Report_Count"].max()
else:
    dt_escal = pd.DataFrame(columns=["DT Nomenclature","location_trust_score"])
customer_monthly_sel = customer_monthly_sel.merge(dt_escal[["DT Nomenclature","location_trust_score"]], left_on="NAME_OF_DT", right_on="DT Nomenclature", how="left")
customer_monthly_sel = customer_monthly_sel.merge(feeder_escal[["Feeder","location_trust_score"]], on="Feeder", how="left", suffixes=("_dt","_feeder"))
customer_monthly_sel["location_trust_score"] = customer_monthly_sel["location_trust_score_dt"].combine_first(customer_monthly_sel["location_trust_score_feeder"]).fillna(0)
customer_monthly_sel["pattern_deviation_score"] = customer_monthly_sel["pattern_deviation_score"].fillna(0)
customer_monthly_sel["zero_counter_score"] = customer_monthly_sel["zero_counter_score"].fillna(0)
customer_monthly_sel["dt_relative_usage_score"] = customer_monthly_sel["dt_relative_usage_score"].fillna(0)
customer_monthly_sel["feeder_billing_efficiency"] = customer_monthly_sel["feeder_billing_efficiency"].fillna(0)
customer_monthly_sel["dt_billing_efficiency"] = customer_monthly_sel["dt_billing_efficiency"].fillna(0)
customer_monthly_sel["theft_probability"] = (
    w_feeder * (1 - customer_monthly_sel["feeder_billing_efficiency"]) +
    w_dt * (1 - customer_monthly_sel["dt_billing_efficiency"]) +
    w_location * customer_monthly_sel["location_trust_score"] +
    w_pattern * customer_monthly_sel["pattern_deviation_score"] +
    w_relative * customer_monthly_sel["dt_relative_usage_score"] +
    w_zero * customer_monthly_sel["zero_counter_score"]
).clip(0,1)
customer_monthly_sel["risk_tier"] = pd.cut(customer_monthly_sel["theft_probability"], bins=[0,0.4,0.7,1.0], labels=["Low","Medium","High"], include_lowest=True)
month_customers = customer_monthly_sel.groupby(["ACCOUNT_NUMBER","METER_NUMBER","CUSTOMER_NAME","ADDRESS","Billing_Type"], as_index=False).agg({
    "billed_kwh":"sum",
    "theft_probability":"mean",
    "pattern_deviation_score":"mean",
    "dt_relative_usage_score":"mean",
    "zero_counter_score":"mean"
})
month_customers = month_customers.rename(columns={"billed_kwh":"billed_kwh_total","theft_probability":"theft_probability_avg"})

# Display customer list
st.subheader(f"Customers under {selected_dt_short} ({selected_feeder_short}, {start_month} to {end_month})")
if selected_dt_short:
    filtered_customers = customer_monthly_sel[customer_monthly_sel["DT_Short_Name"] == selected_dt_short]
    if filtered_customers.empty:
        st.warning("No customers for this DT in selected period.")
    else:
        display_df = filtered_customers.groupby(["ACCOUNT_NUMBER","METER_NUMBER","CUSTOMER_NAME","ADDRESS","Billing_Type"], as_index=False).agg({
            "billed_kwh":"sum",
            "theft_probability":"mean",
            "pattern_deviation_score":"mean",
            "dt_relative_usage_score":"mean",
            "zero_counter_score":"mean"
        }).rename(columns={"billed_kwh":"billed_kwh_total","theft_probability":"theft_probability_avg"})
        display_df = display_df.sort_values("theft_probability_avg", ascending=False)
        st.dataframe(display_df.style.format({
            "billed_kwh_total":"{:.2f}",
            "theft_probability_avg":"{:.3f}",
            "pattern_deviation_score":"{:.3f}",
            "dt_relative_usage_score":"{:.3f}",
            "zero_counter_score":"{:.3f}"
        }), use_container_width=True)

# Customer Heatmap
st.subheader("Theft Probability Heatmap by Customer")
if selected_dt_short:
    try:
        filtered_customers = customer_monthly_sel[customer_monthly_sel["DT_Short_Name"] == selected_dt_short]
        if filtered_customers.empty:
            st.warning("No customer monthly data for this DT.")
        else:
            customer_scores = filtered_customers.groupby("ACCOUNT_NUMBER")["theft_probability"].mean().reset_index()
            customer_order = customer_scores.sort_values("theft_probability", ascending=False)["ACCOUNT_NUMBER"].tolist()
            num_customers = st.number_input("Number of high-risk customers for Heatmap (0 for all)", min_value=0, value=min(10, len(customer_order)), step=1)
            if num_customers > 0:
                chosen = customer_order[:num_customers]
                filtered_for_heatmap = filtered_customers[filtered_customers["ACCOUNT_NUMBER"].isin(chosen)]
            else:
                filtered_for_heatmap = filtered_customers
            pivot_data = filtered_for_heatmap.pivot_table(index="ACCOUNT_NUMBER", columns="month", values="theft_probability", aggfunc="mean").reindex(index=customer_order[:num_customers or None], columns=selected_months)
            if not pivot_data.empty:
                plt.figure(figsize=(10, max(4, len(pivot_data)/2)))
                sns.heatmap(pivot_data, vmin=0, vmax=1, cbar_kws={"label":"Theft Probability"})
                plt.title(f"Theft Probability for {selected_dt_short} ({selected_feeder_short})")
                st.pyplot(plt.gcf())
                plt.close()
            else:
                st.warning("No data to render heatmap.")
    except Exception as e:
        st.error(f"Customer heatmap failed: {e}")

# DT Theft Heatmap
st.subheader("DT Theft Probability Heatmap")
if selected_feeder:
    try:
        dt_filtered = dt_merged_monthly[dt_merged_monthly["Feeder"] == selected_feeder]
        if dt_filtered.empty:
            st.warning("No DT monthly data for this feeder.")
        else:
            dt_scores = dt_filtered.groupby("DT_Short_Name")["dt_billing_efficiency"].mean().reset_index()
            dt_scores["theft_probability"] = 1 - dt_scores["dt_billing_efficiency"]
            order = dt_scores.sort_values("theft_probability", ascending=False)["DT_Short_Name"].tolist()
            dt_pivot = dt_filtered.pivot_table(index="DT_Short_Name", columns="month", values="dt_billing_efficiency", aggfunc="mean").reindex(index=order, columns=selected_months)
            if not dt_pivot.empty:
                plt.figure(figsize=(10, max(4, len(dt_pivot)/2)))
                sns.heatmap(1 - dt_pivot, vmin=0, vmax=1, cbar_kws={"label":"DT Theft Probability"})
                plt.title(f"DT Theft Probability for {selected_feeder_short} ({start_month} to {end_month})")
                st.pyplot(plt.gcf())
                plt.close()
            else:
                st.warning("No DT pivot data.")
    except Exception as e:
        st.error(f"DT heatmap failed: {e}")

# Feeder Summary
st.subheader("Feeder Summary")
try:
    feeder_summary = feeder_merged.copy()
    feeder_summary["Period"] = f"{start_month} to {end_month}"
    st.dataframe(feeder_summary.style.format({
        "feeder_energy_kwh":"{:.2f}",
        "total_billed_kwh":"{:.2f}",
        "feeder_billing_efficiency":"{:.3f}"
    }), use_container_width=True)
except Exception as e:
    st.error(f"Feeder summary failed: {e}")

# DT Summary
st.subheader("DT Summary")
try:
    dt_summary_show = dt_merged.groupby("DT_Short_Name").agg({
        "total_dt_kwh":"sum",
        "total_billed_kwh":"sum",
        "dt_billing_efficiency":"mean"
    }).reset_index()
    st.dataframe(dt_summary_show.style.format({
        "total_dt_kwh":"{:.2f}",
        "total_billed_kwh":"{:.2f}",
        "dt_billing_efficiency":"{:.3f}"
    }), use_container_width=True)
except Exception as e:
    st.error(f"DT summary failed: {e}")

# Export customer list CSV
st.subheader("Export Customer Data")
try:
    if not month_customers.empty:
        csv = month_customers.to_csv(index=False)
        st.download_button(label=f"Download Customer List ({start_month} to {end_month})", data=csv, file_name=f"theft_analysis_{start_month}_to_{end_month}.csv", mime="text/csv")
    else:
        st.info("No customer list to export.")
except Exception as e:
    st.error(f"CSV export failed: {e}")

# Escalations report
st.subheader("Escalations Report (full lookup of 'Account No')")
try:
    cust_scores_avg = customer_monthly_sel.groupby("ACCOUNT_NUMBER", as_index=False)["theft_probability"].mean().rename(columns={"theft_probability":"theft_probability"})
    escal_report_df = generate_escalations_report(ppm_df, ppd_df, escalations_df, cust_scores_avg, months)
    if escal_report_df.empty:
        st.info("Escalations report produced no rows.")
    else:
        if "Theft Probability (avg)" in escal_report_df.columns:
            escal_report_df["Theft Probability (avg)"] = pd.to_numeric(escal_report_df["Theft Probability (avg)"], errors="coerce")
        st.dataframe(escal_report_df.fillna(""), use_container_width=True)
        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
            escal_report_df.to_excel(writer, index=False, sheet_name="Escalations Report")
        st.download_button(label="📥 Download Escalations Report (Excel)", data=towrite.getvalue(), file_name="Escalations_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception as e:
    st.error(f"Failed to generate escalations report: {e}")

# Footer
st.markdown("Built by Elvis Ebenuwah for Ikeja Electric. 2025.")
