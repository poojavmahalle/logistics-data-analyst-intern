# ============================================================
# YUVA INTERN - LOGISTICS DATA ANALYST INTERNSHIP
# WEEK 3: ADVANCED DATA ANALYSIS AND VISUALIZATION
# Dataset: Food Delivery Times
# ============================================================
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Use the cleaned dataset created in Week 2
DATA_PATH = BASE_DIR / "Food_Delivery_Times_Cleaned.csv"

OUTPUT_DIR = BASE_DIR / "week3_outputs"
CHART_DIR = OUTPUT_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD AND INSPECT DATA
# ------------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place Food_Delivery_Times_Cleaned.csv in the "
        "same folder as this Python script, or update DATA_PATH."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("WEEK 3: ADVANCED LOGISTICS DATA ANALYSIS")
print("=" * 60)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

# Remove exact duplicate rows from the analysis copy
eda_df = df.drop_duplicates().copy()

# Convert numeric variables safely
numeric_cols = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Delivery_Time_min"
]

categorical_cols = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type"
]

required_cols = numeric_cols + categorical_cols

missing_cols = [
    col for col in required_cols if col not in eda_df.columns
]

if missing_cols:
    raise ValueError(
        f"Missing expected columns: {missing_cols}"
    )

for col in numeric_cols:
    eda_df[col] = pd.to_numeric(
        eda_df[col], errors="coerce"
    )

for col in categorical_cols:
    eda_df[col] = eda_df[col].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )

# Standard chart appearance
sns.set_theme(style="whitegrid")

# Save charts as high-resolution images
def save_chart(filename):
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# ------------------------------------------------------------
# 3. KEY PERFORMANCE INDICATORS (KPIs)
# ------------------------------------------------------------

total_orders = len(eda_df)

average_delivery_time = (
    eda_df["Delivery_Time_min"].mean()
)

average_distance = eda_df["Distance_km"].mean()

median_delivery_time = (
    eda_df["Delivery_Time_min"].median()
)

print("\n" + "=" * 60)
print("KEY PERFORMANCE INDICATORS")
print("=" * 60)

print("Total records:", total_orders)
print(
    "Average delivery time:",
    round(average_delivery_time, 2), "minutes"
)
print(
    "Median delivery time:",
    round(median_delivery_time, 2), "minutes"
)
print(
    "Average delivery distance:",
    round(average_distance, 2), "km"
)

kpi_summary = pd.DataFrame({
    "Metric": [
        "Total records",
        "Average delivery time (minutes)",
        "Median delivery time (minutes)",
        "Average delivery distance (km)"
    ],
    "Value": [
        total_orders,
        average_delivery_time,
        median_delivery_time,
        average_distance
    ]
})

kpi_summary.to_csv(
    OUTPUT_DIR / "kpi_summary.csv", index=False
)

# ------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

stats_data = eda_df[numeric_cols]

descriptive_stats = stats_data.describe().T
descriptive_stats["median"] = stats_data.median()
descriptive_stats["variance"] = stats_data.var()

print("\nDescriptive statistics:")
print(descriptive_stats.round(2))

descriptive_stats.to_csv(
    OUTPUT_DIR / "descriptive_statistics.csv"
)

# ------------------------------------------------------------
# CHART 1: DELIVERY TIME DISTRIBUTION
# ------------------------------------------------------------

delivery = eda_df["Delivery_Time_min"].dropna()

plt.figure(figsize=(9, 5))

sns.histplot(
    delivery,
    bins=30,
    kde=True,
    color="steelblue"
)

plt.title("Distribution of Food Delivery Times")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Number of Orders")

save_chart("01_delivery_time_distribution.png")

# ------------------------------------------------------------
# CHART 2: DELIVERY TIME BY TRAFFIC LEVEL
# ------------------------------------------------------------

traffic_data = eda_df[
    ["Traffic_Level", "Delivery_Time_min"]
].dropna()

traffic_order = ["Low", "Medium", "High"]

available_traffic = [
    x for x in traffic_order
    if x in traffic_data["Traffic_Level"].unique()
]

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=traffic_data,
    x="Traffic_Level",
    y="Delivery_Time_min",
    order=available_traffic
)

plt.title("Delivery Time by Traffic Level")
plt.xlabel("Traffic Level")
plt.ylabel("Delivery Time (minutes)")

save_chart("02_delivery_time_by_traffic.png")

# ------------------------------------------------------------
# CHART 3: DISTANCE VS DELIVERY TIME
# ------------------------------------------------------------

distance_data = eda_df[
    ["Distance_km", "Delivery_Time_min"]
].dropna()

plt.figure(figsize=(9, 5))

sns.regplot(
    data=distance_data,
    x="Distance_km",
    y="Delivery_Time_min",
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"}
)

plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (minutes)")

save_chart("03_distance_vs_delivery_time.png")

# ------------------------------------------------------------
# CHART 4: CORRELATION HEATMAP
# ------------------------------------------------------------

# Exclude Order_ID because it is an identifier
corr_cols = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
    "Delivery_Time_min"
]

corr_matrix = eda_df[corr_cols].corr()

print("\nCorrelation matrix:")
print(corr_matrix.round(2))

corr_matrix.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    center=0
)

plt.title("Correlation Heatmap of Logistics Variables")

save_chart("04_correlation_heatmap.png")

# ------------------------------------------------------------
# CHART 5: AVERAGE DELIVERY TIME BY TRAFFIC
# ------------------------------------------------------------

traffic_summary = (
    traffic_data.groupby("Traffic_Level")["Delivery_Time_min"]
    .agg(["count", "mean", "median", "std"])
    .reindex(available_traffic)
)

print("\nDelivery time summary by traffic:")
print(traffic_summary.round(2))

traffic_summary.to_csv(
    OUTPUT_DIR / "traffic_summary.csv"
)

traffic_means = traffic_summary["mean"].dropna()

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=traffic_means.index,
    y=traffic_means.values,
    color="orange"
)

ax.bar_label(ax.containers[0], fmt="%.2f")

plt.title("Average Delivery Time by Traffic Level")
plt.xlabel("Traffic Level")
plt.ylabel("Average Delivery Time (minutes)")

save_chart("05_average_delivery_by_traffic.png")

if "High" in traffic_means.index and "Low" in traffic_means.index:
    difference = (
        traffic_means["High"] - traffic_means["Low"]
    )
    print(
        "\nHigh minus Low traffic average:",
        round(difference, 2), "minutes"
    )

# ------------------------------------------------------------
# CHART 6: PREPARATION TIME VS DELIVERY TIME
# ------------------------------------------------------------

preparation_data = eda_df[
    ["Preparation_Time_min", "Delivery_Time_min"]
].dropna()

plt.figure(figsize=(9, 5))

sns.regplot(
    data=preparation_data,
    x="Preparation_Time_min",
    y="Delivery_Time_min",
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"}
)

plt.title("Preparation Time vs Delivery Time")
plt.xlabel("Preparation Time (minutes)")
plt.ylabel("Delivery Time (minutes)")

save_chart("06_preparation_vs_delivery_time.png")

# ------------------------------------------------------------
# CHART 7: DELIVERY TIME BY WEATHER
# ------------------------------------------------------------

weather_data = eda_df[
    ["Weather", "Delivery_Time_min"]
].dropna()

weather_summary = (
    weather_data.groupby("Weather")["Delivery_Time_min"]
    .agg(["count", "mean", "median"])
)

weather_summary.to_csv(
    OUTPUT_DIR / "weather_summary.csv"
)

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=weather_data,
    x="Weather",
    y="Delivery_Time_min"
)

plt.title("Delivery Time by Weather Condition")
plt.xlabel("Weather")
plt.ylabel("Delivery Time (minutes)")
plt.xticks(rotation=20)

save_chart("07_delivery_time_by_weather.png")

# ------------------------------------------------------------
# CHART 8: DELIVERY TIME BY TIME OF DAY
# ------------------------------------------------------------

time_data = eda_df[
    ["Time_of_Day", "Delivery_Time_min"]
].dropna()

time_summary = (
    time_data.groupby("Time_of_Day")["Delivery_Time_min"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)

print("\nDelivery time by time of day:")
print(time_summary.round(2))

time_summary.to_csv(
    OUTPUT_DIR / "time_of_day_summary.csv"
)

plt.figure(figsize=(9, 5))

ax = sns.barplot(
    data=time_summary.reset_index(),
    x="Time_of_Day",
    y="mean",
    color="mediumseagreen"
)

ax.bar_label(ax.containers[0], fmt="%.2f")

plt.title("Average Delivery Time by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Average Delivery Time (minutes)")
plt.xticks(rotation=20)

save_chart("08_delivery_time_by_time_of_day.png")

# ------------------------------------------------------------
# CHART 9: DELIVERY TIME BY VEHICLE TYPE
# ------------------------------------------------------------

vehicle_data = eda_df[
    ["Vehicle_Type", "Delivery_Time_min"]
].dropna()

vehicle_summary = (
    vehicle_data.groupby("Vehicle_Type")["Delivery_Time_min"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
)

print("\nDelivery time by vehicle type:")
print(vehicle_summary.round(2))

vehicle_summary.to_csv(
    OUTPUT_DIR / "vehicle_type_summary.csv"
)

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=vehicle_data,
    x="Vehicle_Type",
    y="Delivery_Time_min"
)

plt.title("Delivery Time by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Delivery Time (minutes)")
plt.xticks(rotation=20)

save_chart("09_delivery_time_by_vehicle.png")

# ------------------------------------------------------------
# CHART 10: COURIER EXPERIENCE VS DELIVERY TIME
# ------------------------------------------------------------

experience_data = eda_df[
    ["Courier_Experience_yrs", "Delivery_Time_min"]
].dropna()

plt.figure(figsize=(9, 5))

sns.regplot(
    data=experience_data,
    x="Courier_Experience_yrs",
    y="Delivery_Time_min",
    scatter_kws={"alpha": 0.5},
    line_kws={"color": "red"}
)

plt.title("Courier Experience vs Delivery Time")
plt.xlabel("Courier Experience (years)")
plt.ylabel("Delivery Time (minutes)")

save_chart("10_courier_experience_vs_delivery_time.png")

# ------------------------------------------------------------
# 5. EXPORT EDA DATA
# ------------------------------------------------------------

eda_df.to_csv(
    OUTPUT_DIR / "Food_Delivery_Times_EDA.csv",
    index=False
)

print("\n" + "=" * 60)
print("WEEK 3 ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Charts saved in:", CHART_DIR)
print("Summary files saved in:", OUTPUT_DIR)