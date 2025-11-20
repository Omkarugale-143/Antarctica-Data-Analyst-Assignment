
# Antarctica – Data Analyst Assignment
# By : Omkar Ugale


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


# LOAD DATASET


DATA_PATH = "C:/Users/OMKAR/Downloads/Data Analyst Practice/Data/owid-co2-data.csv"

df_raw = pd.read_csv(DATA_PATH, low_memory=False)
print("Raw dataset loaded.")
print("Shape:", df_raw.shape)


# DATA CLEANING


# Keep only country rows (ISO code length = 3)
df = df_raw[df_raw["iso_code"].str.len() == 3].copy()

# Filter years from 1980 onwards
df = df[df["year"] >= 1980]

# Select relevant columns
cols = [
    "country", "iso_code", "year",
    "co2", "co2_per_capita",
    "primary_energy_consumption", "energy_per_capita",
    "co2_per_unit_energy",   # carbon intensity
    "population"
]

df = df[cols]

# Convert numeric columns
for col in ["year", "co2", "co2_per_capita",
            "primary_energy_consumption", "energy_per_capita",
            "co2_per_unit_energy", "population"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows missing critical values
df = df.dropna(subset=["co2", "primary_energy_consumption", "energy_per_capita"])
print("Cleaned dataset shape:", df.shape)

# Q1 — GLOBAL EMISSIONS PROJECTION (NEXT 5 YEARS)


print("\n================ Q1: Global CO2 Emission Projection ================\n")
df_world = df_raw[df_raw["country"] == "World"]
df_world = df_world[df_world["year"] >= 1980].dropna(subset=["co2"])

X = df_world["year"].values.reshape(-1, 1)
y = df_world["co2"].values

model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)

# Confidence interval
y_err = y - pred
y_std = np.std(y_err)
conf = 1.96 * y_std

future_years = np.arange(df_world["year"].max() + 1, df_world["year"].max() + 6)
future_preds = model.predict(future_years.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(df_world["year"], y, label="Actual CO₂", linewidth=2)
plt.plot(df_world["year"], pred, label="Trend Line", linestyle='--')

plt.fill_between(df_world["year"], pred - conf, pred + conf, alpha=0.2, label="95% Confidence")

plt.plot(future_years, future_preds, "ro--", label="Forecast")
plt.title("Global CO₂ Emission Trend + 5-Year Forecast")
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions")
plt.legend()
plt.show()

# Q2 — Cloud Region Carbon Intensity (Advanced Multi-Year Visuals)


print("\n================Q2 — Cloud Region Carbon Intensity (Advanced Multi-Year Visuals) ================\n")

regions = [
    "United States", "Canada", "Brazil",
    "Ireland", "United Kingdom", "Germany", "France", "Netherlands",
    "India", "Japan", "South Korea", "Singapore", "Australia",
    "United Arab Emirates"
]

df_regions = df[df["country"].isin(regions)].copy()

max_year = df_regions["year"].max()
years = list(range(max_year - 4, max_year + 1))

df_5yrs = df_regions[df_regions["year"].isin(years)]
pivot_df = df_5yrs.pivot(index="country", columns="year",
                         values="co2_per_unit_energy").reindex(regions)

print("Years analyzed:", years)
print("\n5-Year Carbon Intensity Table:")
print(pivot_df)

plt.figure(figsize=(14, 7))
bar_width = 0.12
x = np.arange(len(regions))

for i, year in enumerate(years):
    plt.bar(x + i * bar_width, pivot_df[year], bar_width, label=str(year))

plt.xticks(x + bar_width*2, regions, rotation=45)
plt.ylabel("CO₂ per Unit Energy")
plt.title("Carbon Intensity for Cloud Regions (Past 5 Years - Grouped Bar Chart)")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
sns.heatmap(pivot_df, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Carbon Intensity Over 5 Years")
plt.show()

# Q3 — Data Center vs Nations (Advanced Benchmark Visuals)
print("======Q3 — Data Center vs Nations (Advanced Benchmark Visuals)=========")

df_world_latest = df_world[df_world["year"] == df_world["year"].max()]
world_year = int(df_world_latest["year"].iloc[0])
world_co2 = df_world_latest["co2"].iloc[0]

data_center_co2 = 0.02 * world_co2

print("World Year:", world_year)
print("World CO₂:", world_co2)
print("Estimated Data Center CO₂ (2%):", data_center_co2)

countries = ["United States", "Canada", "Brazil",
    "Ireland", "United Kingdom", "Germany", "France", "Netherlands",
    "India", "Japan", "South Korea", "Singapore", "Australia",
    "United Arab Emirates"]
df_big = df[df["country"].isin(countries)]
df_big_latest = df_big[df_big["year"] == world_year][["country", "co2"]]

compare_df = pd.concat([
    df_big_latest,
    pd.DataFrame([{"country": "Data Centers (Est.)", "co2": data_center_co2}])
]).sort_values("co2")

print("\nComparison Table:")
print(compare_df)

# Donut chart
sizes = [data_center_co2, world_co2 - data_center_co2]
labels = ["Data Centers", "Rest of World"]

plt.figure(figsize=(7,7))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", pctdistance=0.85)
centre = plt.Circle((0,0), 0.60, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre)
plt.title("Data Centers' Share of Global CO₂ (Donut Chart)")
plt.tight_layout()
plt.show()

# Horizontal bar chart
plt.figure(figsize=(10,6))
plt.barh(compare_df["country"], compare_df["co2"])
plt.xlabel("CO₂ Emissions (Mt)")
plt.title("Data Center CO₂ vs Major Countries")
plt.tight_layout()
plt.show()

print("Data Analyzed Successfully!!")