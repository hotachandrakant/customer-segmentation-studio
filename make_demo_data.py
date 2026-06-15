"""
make_demo_data.py
-----------------
Creates a DEMO csv that matches the EXACT schema of the Kaggle
"Mall Customer Segmentation Data" dataset, so this project runs out-of-the-box.
The data is generated with 5 natural clusters, exactly like the real dataset.

>>> Replace this with the REAL dataset for your submission:
    https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
    Download "Mall_Customers.csv" and put it in this same data/ folder.
    The loader automatically prefers the real file if it is present.

Usage:
    python make_demo_data.py
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# 5 natural customer groups (income k$, spending score 1-100), like the real data
groups = [
    # (n, income_mean, income_sd, spend_mean, spend_sd, age_mean, age_sd)
    (40,  25,  8,  20, 10, 45, 12),   # low income, low spending
    (35,  25,  8,  78,  9, 25,  5),   # low income, high spending
    (80,  55, 10,  50, 12, 43, 14),   # average income, average spending (largest)
    (25,  85, 12,  18,  9, 42, 10),   # high income, low spending
    (40,  85, 12,  82,  9, 33,  6),   # high income, high spending (target)
]

rows, cid = [], 1
for n, im, isd, sm, ssd, am, asd in groups:
    for _ in range(n):
        income = int(np.clip(RNG.normal(im, isd), 15, 140))
        spend = int(np.clip(RNG.normal(sm, ssd), 1, 100))
        age = int(np.clip(RNG.normal(am, asd), 18, 70))
        rows.append({
            "CustomerID": cid,
            "Gender": str(RNG.choice(["Male", "Female"], p=[0.45, 0.55])),
            "Age": age,
            "Annual Income (k$)": income,
            "Spending Score (1-100)": spend,
        })
        cid += 1

df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
df["CustomerID"] = range(1, len(df) + 1)   # renumber after shuffle

df.to_csv("data/sample_demo_data.csv", index=False)
print(f"Demo data written: data/sample_demo_data.csv ({len(df)} rows)")
print(df.head())
print("\nNOTE: replace with the real Kaggle 'Mall_Customers.csv' for submission.")
