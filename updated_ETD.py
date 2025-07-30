import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate data for 4 feeders, 5-9 DTs each, 7-25 buildings per DT (2 weeks)
np.random.seed(42)
dates = pd.date_range(start="2025-07-01", end="2025-07-14", freq="h")
n_hours = len(dates)

# Define building types and consumption ranges (kWh/day)
building_types = {
    "Mall": (500, 1000, 1, 2),  # Min, max, min_count, max_count
    "Hotel": (200, 500, 1, 3),
    "Office": (100, 300, 2, 5),
    "Apartment": (20, 50, 2, 5),
    "Bungalow": (5, 15, 3, 10)
}

# Simulate 4 feeders, each with 5-9 DTs
feeders = ["Fakale", "Oshodi", "Ikeja_Central", "Alausa"]
dt_per_feeder = {f: np.random.randint(5, 10) for f in feeders}  # 5-9 DTs per feeder
dts = []
dt_to_street = {}
dt_to_feeder = {}
for feeder in feeders:
    for i in range(1, dt_per_feeder[feeder] + 1):
        dt_id = f"DT_{feeder}_{i}"
        street_id = f"Street_{dt_id}"
        dts.append(dt_id)
        dt_to_street[dt_id] = street_id
        dt_to_feeder[dt_id] = feeder

# Simulate buildings per DT
buildings = []
for dt in dts:
    street = dt_to_street[dt]
    n_buildings = np.random.randint(10, 26)
    type_counts = {t: np.random.randint(c[2], c[3] + 1) for t, c in building_types.items()}
    total_types = sum(type_counts.values())
    if total_types > n_buildings:
        type_counts = {t: int(c * n_buildings / total_types) for t, c in type_counts.items()}
        total_types = sum(type_counts.values())
    while total_types < n_buildings:
        type_counts["Bungalow"] += 1
        total_types += 1
    for btype, count in type_counts.items():
        for i in range(1, count + 1):
            buildings.append((f"{btype}_{street}_{i}", btype, street, dt))

# Simulate hourly data
data = []
for bid, btype, street, dt in buildings:
    low, high = building_types[btype][:2]
    daily_avg = np.random.uniform(low, high)
    hourly_base = daily_avg / 24
    usage = [max(0, hourly_base * np.random.uniform(0.9, 1.1)) for _ in range(n_hours)]
    payment_history = 0.0 if bid not in [f"Mall_{dt_to_street[dt]}_1" for dt in dts] + [f"Bungalow_{dt_to_street[dt]}_1" for dt in dts] else np.random.uniform(0.8, 1.0)
    location_trust = 0.3 if dt_to_street[dt] not in [f"Street_DT_{f}_1" for f in feeders] else 0.7
    phase_current = np.random.normal(daily_avg / 24 * 10, 1.5, n_hours)
    data.append(pd.DataFrame({
        "timestamp": dates,
        "feeder_id": dt_to_feeder[dt],
        "dt_id": dt,
        "street_id": street,
        "building_id": bid,
        "building_type": btype,
        "phase_current": phase_current,
        "neutral_current": phase_current.copy(),
        "usage_kwh": usage,
        "payment_history": payment_history,
        "location_trust": location_trust
    }))

data = pd.concat(data, ignore_index=True)

# Simulate persistent theft
theft_buildings = {f"Mall_{dt_to_street[dt]}_1": (pd.to_datetime("2025-07-03"), pd.to_datetime("2025-07-07")) for dt in dts if f"Mall_{dt_to_street[dt]}_1" in [b[0] for b in buildings]}
theft_buildings.update({f"Bungalow_{dt_to_street[dt]}_1": (pd.to_datetime("2025-07-02"), pd.to_datetime("2025-07-08")) for dt in dts if f"Bungalow_{dt_to_street[dt]}_1" in [b[0] for b in buildings]})
for bid, (start, end) in theft_buildings.items():
    mask = (data["building_id"] == bid) & (data["timestamp"] >= start) & (data["timestamp"] <= end)
    data.loc[mask, "usage_kwh"] *= np.random.uniform(0.3, 0.5)
    data.loc[mask, "neutral_current"] *= 0.5

# Calculate daily data
data["date"] = data["timestamp"].dt.date
daily_data = data.groupby(["date", "feeder_id", "dt_id", "street_id", "building_id", "building_type"]).agg({
    "usage_kwh": "sum",
    "payment_history": "mean",
    "location_trust": "mean"
}).reset_index()
daily_data["expected_kwh"] = daily_data.groupby("building_id")["usage_kwh"].transform("mean")

# Simulate DT-level data
dt_data = daily_data.groupby(["date", "feeder_id", "dt_id"]).agg({"building_id": "count", "building_type": lambda x: list(x)}).reset_index()
dt_data["max_expected_kwh"] = dt_data.apply(
    lambda row: sum(building_types[btype][1] for btype in row["building_type"]), axis=1
)
dt_data["supplied_kwh"] = dt_data["max_expected_kwh"] * np.random.uniform(1.1, 1.2)
dt_data["metered_kwh"] = daily_data.groupby(["date", "feeder_id", "dt_id"])["usage_kwh"].sum().reset_index()["usage_kwh"]
dt_data["dt_ratio"] = dt_data["supplied_kwh"] / dt_data["metered_kwh"]
dt_data["dt_score"] = ((dt_data["dt_ratio"] - 1.1) / (2 - 1.1)).clip(0, 1)
dt_data["financial_loss_naira"] = (dt_data["supplied_kwh"] - dt_data["metered_kwh"]) * 209.5

# Merge DT data with left join
daily_data = daily_data.merge(dt_data[["date", "feeder_id", "dt_id", "dt_score"]], on=["date", "feeder_id", "dt_id"], how="left")
daily_data["dt_score"] = daily_data["dt_score"].fillna(0)

# Streamlit dashboard
st.title("IKEDC Energy Theft Detection Dashboard")
st.write("Detect high-risk buildings (flagged ≥3 days), prioritize DTs, estimate savings (₦209.5/kWh)")

# Weight adjustment sliders
st.subheader("Adjust Weights for Theft Probability")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    dt_weight = st.slider("DT Score Weight", 0.0, 1.0, 0.45, 0.05)
with col2:
    consumption_weight = st.slider("Consumption Score Weight", 0.0, 1.0, 0.2, 0.05)
with col3:
    payment_weight = st.slider("Payment History Weight", 0.0, 1.0, 0.15, 0.05)
with col4:
    deviation_weight = st.slider("Pattern Deviation Weight", 0.0, 1.0, 0.15, 0.05)
with col5:
    location_weight = st.slider("Location Trust Weight", 0.0, 1.0, 0.05, 0.05)

# Normalize weights
total_weight = dt_weight + consumption_weight + payment_weight + deviation_weight + location_weight
if total_weight > 0:
    dt_weight /= total_weight
    consumption_weight /= total_weight
    payment_weight /= total_weight
    deviation_weight /= total_weight
    location_weight /= total_weight
else:
    st.error("Total weight cannot be zero. Using default weights.")
    dt_weight, consumption_weight, payment_weight, deviation_weight, location_weight = 0.45, 0.2, 0.15, 0.15, 0.05

# Calculate weighted features
daily_data["consumption_score"] = (1 - daily_data["usage_kwh"] / daily_data["expected_kwh"]).clip(0, 1)
daily_data["pattern_deviation"] = abs(daily_data["usage_kwh"] - daily_data["expected_kwh"]) / daily_data["expected_kwh"]
daily_data["theft_probability"] = (
    dt_weight * daily_data["dt_score"] +
    consumption_weight * daily_data["consumption_score"] +
    payment_weight * daily_data["payment_history"] +
    deviation_weight * daily_data["pattern_deviation"] +
    location_weight * daily_data["location_trust"]
).clip(0, 1)

# Train Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)
features = daily_data[["theft_probability", "dt_score", "consumption_score"]]
daily_data["is_theft"] = model.fit_predict(features) == -1

# Filter by feeder, DT, date range, building type, probability
st.subheader("Filter Data for Building-Level Analysis")
col1, col2, col3, col4 = st.columns(4)
with col1:
    feeder = st.selectbox("Select Feeder", feeders)
with col2:
    dt = st.selectbox("Select DT", [dt for dt in dts if dt_to_feeder[dt] == feeder])
with col3:
    date_range = st.date_input("Select Date Range", value=(pd.to_datetime("2025-07-01"), pd.to_datetime("2025-07-14")), min_value=pd.to_datetime("2025-07-01"), max_value=pd.to_datetime("2025-07-14"))
with col4:
    building_type = st.selectbox("Building Type", ["All"] + list(building_types.keys()))
min_prob = st.slider("Minimum Theft Probability", 0.0, 1.0, 0.0)

# Convert date range to datetime.date for filtering
start_date, end_date = date_range
filtered_data = daily_data[(daily_data["feeder_id"] == feeder) & (daily_data["dt_id"] == dt) & (daily_data["date"] >= start_date) & (daily_data["date"] <= end_date)]
if building_type != "All":
    filtered_data = filtered_data[filtered_data["building_type"] == building_type]
filtered_data = filtered_data[filtered_data["theft_probability"] >= min_prob]

# Display filtered data (average theft probability over date range)
st.subheader(f"Average Theft Probability for {dt} ({feeder}, {start_date} to {end_date})")
agg_data = filtered_data.groupby(["building_id", "building_type"])["theft_probability"].mean().reset_index()
is_theft_data = filtered_data.groupby(["building_id", "building_type"])["is_theft"].sum().reset_index()
agg_data = agg_data.merge(is_theft_data, on=["building_id", "building_type"], how="left")
agg_data["is_theft"] = agg_data["is_theft"] > 0
st.dataframe(agg_data[["building_id", "building_type", "theft_probability", "is_theft"]].round(3))

# Savings estimates
st.subheader("Potential Savings Estimates (₦209.5/kWh)")
daily_savings = dt_data[(dt_data["feeder_id"] == feeder) & (dt_data["date"] >= start_date) & (dt_data["date"] <= end_date)]["financial_loss_naira"].sum() / (end_date - start_date).days if (end_date - start_date).days > 0 else 0
total_savings = dt_data[(dt_data["date"] >= start_date) & (dt_data["date"] <= end_date)]["financial_loss_naira"].sum()
avg_daily_savings = total_savings / (end_date - start_date).days if (end_date - start_date).days > 0 else total_savings
monthly_savings = avg_daily_savings * 30
yearly_savings = avg_daily_savings * 365
st.write(f"Average Daily Savings ({start_date} to {end_date}): ₦{daily_savings:,.2f}")
st.write(f"Total Savings for Period: ₦{total_savings:,.2f}")
st.write(f"Estimated Monthly Savings (30 days): ₦{monthly_savings:,.2f}")
st.write(f"Estimated Yearly Savings (365 days): ₦{yearly_savings:,.2f}")

# DT-level heatmap
st.subheader(f"Theft Probability Heatmap for {dt} (White to Red)")
pivot_data = filtered_data.groupby(["building_id", "date"])["theft_probability"].mean().reset_index()
pivot_data = pivot_data.pivot(index="building_id", columns="date", values="theft_probability")
if not pivot_data.empty:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_data, cmap="YlOrRd", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Theft Probability"})
    ax.set_xlabel("Date")
    ax.set_ylabel("Building ID")
    ax.set_title(f"Theft Probability for {dt} ({feeder}, {start_date} to {end_date})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.write("No data available for the selected filters.")

# DT-ranking heatmap
st.subheader(f"DT-Ranking Heatmap for {feeder} (White to Red)")
dt_pivot = daily_data[(daily_data["feeder_id"] == feeder) & (daily_data["date"] >= start_date) & (daily_data["date"] <= end_date)].groupby(["date", "dt_id"])["theft_probability"].mean().reset_index()
dt_pivot = dt_pivot.pivot(index="dt_id", columns="date", values="theft_probability")
dt_order = dt_pivot.mean(axis=1).sort_values(ascending=False).index
dt_pivot = dt_pivot.loc[dt_order]
if not dt_pivot.empty:
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(dt_pivot, cmap="YlOrRd", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Average Theft Probability"})
    ax.set_xlabel("Date")
    ax.set_ylabel("DT ID")
    ax.set_title(f"DT-Ranking Heatmap for {feeder} ({start_date} to {end_date})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.write("No DT data available for the selected filters.")

# Financial loss bar chart
st.subheader(f"Estimated Financial Loss per DT ({feeder}, ₦)")
loss_data = dt_data[(dt_data["feeder_id"] == feeder) & (dt_data["date"] >= start_date) & (dt_data["date"] <= end_date)].groupby("dt_id")["financial_loss_naira"].sum()
loss_data = loss_data.reindex(dt_order, fill_value=0)
if not loss_data.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    loss_data.plot(kind="bar", color=["#FF6B6B" if i < 2 else "#FFADAD" for i in range(len(loss_data))], ax=ax)
    ax.set_xlabel("DT ID")
    ax.set_ylabel("Loss (₦)")
    ax.set_title(f"Estimated Financial Loss ({feeder}, {start_date} to {end_date}, ₦209.5/kWh)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.write("No financial loss data available for the selected filters.")

# Summary of high-risk buildings (flagged ≥3 days)
st.subheader(f"High-Risk Buildings Summary for {feeder} (Flagged ≥3 Days)")
theft_count = daily_data[(daily_data["feeder_id"] == feeder) & (daily_data["date"] >= start_date) & (daily_data["date"] <= end_date) & (daily_data["is_theft"])].groupby(["dt_id", "building_id"]).size().reset_index(name="days_flagged")
high_risk = theft_count[theft_count["days_flagged"] >= 3]
st.write(f"Total High-Risk Buildings (Flagged ≥3 Days): {len(high_risk)}")
if len(high_risk) > 0:
    for _, row in high_risk.iterrows():
        st.write(f"{row['dt_id']} - {row['building_id']}: Flagged for {row['days_flagged']} days")
else:
    st.write("No buildings flagged for ≥3 days in the selected date range.")