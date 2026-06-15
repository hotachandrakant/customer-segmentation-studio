"""
utils.py
--------
Shared helpers: a robust loader for the Kaggle "Mall Customer Segmentation"
dataset (Mall_Customers.csv) and small utilities.

The loader:
  - prefers the REAL Kaggle file if present (data/Mall_Customers.csv)
  - falls back to the demo file (data/sample_demo_data.csv)
  - normalises column names: 'Genre' -> 'Gender', long names -> Income / Spending
  - handles small numbers of missing values (some Kaggle versions include them)
"""

import os
import pandas as pd

CANDIDATES = [
    "data/Mall_Customers.csv",
    "data/mall_customers.csv",
    "data/customer_segmentation_1000.csv",
    "data/sample_demo_data.csv",
]

# Map the many real-world column spellings to clean internal names
RENAME = {
    "Genre": "Gender",
    "Annual Income (k$)": "Income",
    "Annual_income": "Income",
    "Annual Income (K$)": "Income",
    "Spending Score (1-100)": "Spending",
    "Spending_score": "Spending",
    "Spending Score (1–100)": "Spending",
}


def find_data_file():
    for path in CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No dataset in data/. Run `python make_demo_data.py` or download "
        "'Mall_Customers.csv' from Kaggle into data/."
    )


def load_customers():
    """Load + clean the mall-customers data; returns (DataFrame, source_filename)."""
    path = find_data_file()
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})

    # Standardise gender text
    if "Gender" in df:
        df["Gender"] = (df["Gender"].astype(str).str.strip().str.title()
                        .replace({"M": "Male", "F": "Female"}))

    # Handle missing values: numeric -> median, gender -> mode
    for col in ["Age", "Income", "Spending"]:
        if col in df and df[col].isna().any():
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
    if "Gender" in df and df["Gender"].isna().any():
        df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])

    df = df.drop_duplicates().reset_index(drop=True)
    return df, os.path.basename(path)
