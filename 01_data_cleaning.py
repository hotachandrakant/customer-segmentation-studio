"""
01_data_cleaning.py
-------------------
Loads the raw Mall Customers data, runs data-quality checks, cleans it, and
writes a tidy data/customers_clean.csv used by the segmentation + dashboard.

Usage:
    python 01_data_cleaning.py
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_customers

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
os.makedirs("outputs", exist_ok=True)

df, source = load_customers()

print("=" * 55)
print(f"  DATA CLEANING REPORT   (source: {source})")
print("=" * 55)
print(f"Rows     : {len(df)}")
print(f"Columns  : {list(df.columns)}")
print("\nMissing values per column:")
miss = df.isna().sum()
print(miss[miss > 0] if miss.sum() else "  none")

print("\nNumeric summary:")
print(df[["Age", "Income", "Spending"]].describe().round(1))

print("\nGender split:")
print(df["Gender"].value_counts())

# ---------------- EDA charts ----------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, col in zip(ax, ["Age", "Income", "Spending"]):
    sns.histplot(df[col], kde=True, ax=a, color="#2563eb")
    a.set_title(f"Distribution of {col}", fontweight="bold")
plt.tight_layout(); plt.savefig("outputs/01_distributions.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(data=df, x="Income", y="Spending", hue="Gender", ax=ax[0],
                palette="Set1", alpha=0.8)
ax[0].set_title("Income vs Spending (by Gender)", fontweight="bold")
sns.boxplot(data=df, x="Gender", y="Spending", ax=ax[1], palette="Set2",
            hue="Gender", legend=False)
ax[1].set_title("Spending Score by Gender", fontweight="bold")
plt.tight_layout(); plt.savefig("outputs/02_eda_gender.png"); plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(df[["Age", "Income", "Spending"]].corr(), annot=True,
            cmap="RdBu_r", center=0, fmt=".2f")
plt.title("Correlation: Age / Income / Spending", fontweight="bold")
plt.tight_layout(); plt.savefig("outputs/03_correlation.png"); plt.close()

df.to_csv("outputs/customers_clean.csv", index=False)
print("\nSaved EDA charts (01-03) + cleaned dataset -> outputs/customers_clean.csv")
