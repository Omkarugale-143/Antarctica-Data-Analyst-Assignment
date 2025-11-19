
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

df_world = df_raw[df_raw["country"] == "World"].copy()
df_world = df_world[df_world["year"] >= 1980]
df_world = df_world.dropna(subset=["co2"])

X = df_world["year"].values.reshape(-1, 1)
y = df_world["co2"].values

model = LinearRegression()
model.fit(X, y)

future_years = np.arange(df_world["year"].max() + 1, df_world["year"].max() + 6)
future_preds = model.predict(future_years.reshape(-1, 1))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df_world["year"], df_world["co2"], label="Historical CO₂")
plt.plot(future_years, future_preds, "r--", label="Projected CO₂ (Next 5 Years)")
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions (million tonnes)")
plt.title("Global CO₂ Emissions Projection (Q1)")
plt.legend()
plt.tight_layout()
plt.show()

print("Projected CO2 emissions (Q1):")
print(pd.DataFrame({"year": future_years, "projected_co2": future_preds}))



# Q2 — CLOUD REGION CARBON INTENSITY COMPARISON


print("\n================ Q2: Cloud Region Carbon Intensity ================\n")

regions = ["United States", "China", "India", "Germany", "Singapore", "Ireland"]
df_regions = df[df["country"].isin(regions)].copy()

latest_year = df_regions["year"].max()
df_latest = df_regions[df_regions["year"] == latest_year].dropna(subset=["co2_per_unit_energy"])

df_latest["carbon_intensity"] = df_latest["co2_per_unit_energy"]

plt.figure(figsize=(10, 5))
plt.bar(df_latest["country"], df_latest["carbon_intensity"])
plt.xticks(rotation=45)
plt.ylabel("CO₂ per unit energy")
plt.xlabel("Cloud Region")
plt.title(f"Carbon Intensity of Cloud Regions ({latest_year}) (Q2)")
plt.tight_layout()
plt.show()

print(df_latest[["country", "carbon_intensity"]])



# Q3 — DATA CENTER CO2 VS NATIONAL EMISSIONS


print("\n================ Q3: Data Center CO2 vs Countries ================\n")

df_world_latest = df_world[df_world["year"] == df_world["year"].max()]
world_year = int(df_world_latest["year"].iloc[0])
world_co2 = df_world_latest["co2"].iloc[0]

data_center_co2 = 0.02 * world_co2  # Assuming 2% estimate

# Pie chart
labels = ["Data Centers (2% Est.)", "Rest of World"]
sizes = [data_center_co2, world_co2 - data_center_co2]

plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
plt.title(f"Data Center Share of Global CO₂ ({world_year}) (Q3)")
plt.show()

# Compare to major countries
countries = ["United States", "China", "India", "Germany"]
df_countries = df[df["country"].isin(countries)]
df_countries_latest = df_countries[df_countries["year"] == world_year][["country", "co2"]]

comparison = pd.concat([
    df_countries_latest,
    pd.DataFrame([{"country": "Data Centers (Est.)", "co2": data_center_co2}])
])

comparison = comparison.sort_values("co2", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(comparison["country"], comparison["co2"])
plt.xticks(rotation=45)
plt.ylabel("CO₂ Emissions (Mt)")
plt.title(f"Data Centers vs Major Countries ({world_year}) (Q3)")
plt.show()

print(comparison)


# ============================================================
# Q4 — ESG REPORTING BLIND SPOTS


print("\n================ Q4: ESG Reporting Blind Spots ================\n")

missing_fraction = df_raw.isna().mean().sort_values(ascending=False)
print("\nTop missing columns:\n", missing_fraction.head(20))

df_intensity = df_raw[df_raw["iso_code"].str.len() == 3]
df_intensity = df_intensity[df_intensity["year"] >= 1980]

df_intensity["missing_co2_intensity"] = df_intensity["co2_per_unit_energy"].isna()

missing_countries = df_intensity[df_intensity["missing_co2_intensity"] == True]["country"].unique()

print("\nCountries missing CO₂ intensity data:", len(missing_countries))
print(missing_countries[:20])



# Q5 — SUSTAINABLE IT STRATEGIES (SUPPORTING CHART)


print("\n================ Q5: Sustainable IT Strategies ================\n")

regions_trend = ["United States", "China", "India", "Germany"]
df_trend = df[df["country"].isin(regions_trend)].copy()

plt.figure(figsize=(10, 6))
for country in regions_trend:
    df_temp = df_trend[df_trend["country"] == country]
    plt.plot(df_temp["year"], df_temp["co2_per_unit_energy"], label=country)

plt.xlabel("Year")
plt.ylabel("CO₂ per unit energy")
plt.title("Carbon Intensity Trends in Major Regions (Q5)")
plt.legend()
plt.tight_layout()
plt.show()

print("Q5 chart completed. Use this trend for strategic recommendations.")

# ============================================================
# END OF SCRIPT
# ============================================================
print("\nAnalysis Completed Successfully.")
