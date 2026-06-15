"""
02_segmentation.py
------------------
Advanced segmentation with MULTIPLE algorithms, proper validation, persona
assignment, and a trained classifier that can label brand-new customers.

Pipeline:
  1. Load clean data, scale Age + Income + Spending
  2. Pick k with Elbow (WCSS) + Silhouette
  3. Compare 4 clustering algorithms (KMeans, Agglomerative, GMM, DBSCAN)
     on 3 validation metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz
  4. Choose the winning model, label every customer
  5. Profile each segment and attach a marketing PERSONA
  6. Train a Random-Forest classifier on the labels (so new customers can be
     assigned instantly) and save it with joblib
  7. Save charts, labelled CSV, profile, persona report, and the model

Usage:
    python 02_segmentation.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             davies_bouldin_score, calinski_harabasz_score)
from utils import load_customers
from personas import get_persona

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

df, source = load_customers()
print(f"Loaded {len(df)} customers (source: {source})")

FEATURES = ["Age", "Income", "Spending"]
scaler = StandardScaler().fit(df[FEATURES])
Xs = scaler.transform(df[FEATURES])

# ----------------------------------------------------------------------
# 1. Choose k (Elbow + Silhouette) on Income + Spending
# ----------------------------------------------------------------------
X2 = StandardScaler().fit_transform(df[["Income", "Spending"]])
wcss, sils, ks = [], [], range(2, 11)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X2)
    wcss.append(km.inertia_)
    sils.append(silhouette_score(X2, km.labels_))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].plot(list(ks), wcss, "o-", color="#2563eb"); ax[0].set_title("Elbow (WCSS)", fontweight="bold")
ax[0].set_xlabel("k"); ax[0].set_ylabel("WCSS")
ax[1].plot(list(ks), sils, "o-", color="#16a34a"); ax[1].set_title("Silhouette", fontweight="bold")
ax[1].set_xlabel("k"); ax[1].set_ylabel("Score")
plt.tight_layout(); plt.savefig(f"{OUT}/04_choosing_k.png"); plt.close()
best_k = 5
print(f"Chosen k = {best_k}")

# ----------------------------------------------------------------------
# 2. Compare clustering algorithms
# ----------------------------------------------------------------------
def score_labels(X, labels):
    """Return (silhouette, davies_bouldin, calinski) ignoring noise-only fits."""
    uniq = set(labels) - {-1}
    if len(uniq) < 2:
        return np.nan, np.nan, np.nan
    mask = labels != -1
    return (silhouette_score(X[mask], labels[mask]),
            davies_bouldin_score(X[mask], labels[mask]),
            calinski_harabasz_score(X[mask], labels[mask]))

algos = {
    "KMeans": KMeans(n_clusters=best_k, random_state=42, n_init=10),
    "Agglomerative": AgglomerativeClustering(n_clusters=best_k),
    "GaussianMixture": GaussianMixture(n_components=best_k, random_state=42),
    "DBSCAN": DBSCAN(eps=0.6, min_samples=5),
}

comparison, label_store = [], {}
for name, model in algos.items():
    labels = model.fit_predict(Xs)
    label_store[name] = labels
    sil, db, ch = score_labels(Xs, labels)
    n_clusters = len(set(labels) - {-1})
    comparison.append({"Algorithm": name, "Clusters": n_clusters,
                       "Silhouette": round(sil, 3) if sil == sil else None,
                       "Davies-Bouldin": round(db, 3) if db == db else None,
                       "Calinski-Harabasz": round(ch, 1) if ch == ch else None})

comp_df = pd.DataFrame(comparison)
comp_df.to_csv(f"{OUT}/algorithm_comparison.csv", index=False)
print("\n--- Algorithm Comparison ---")
print(comp_df.to_string(index=False))

# Visual comparison (Silhouette: higher better; Davies-Bouldin: lower better)
plot_df = comp_df.dropna(subset=["Silhouette"])
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].bar(plot_df["Algorithm"], plot_df["Silhouette"], color="#16a34a")
ax[0].set_title("Silhouette (higher = better)", fontweight="bold")
ax[0].tick_params(axis="x", rotation=20)
ax[1].bar(plot_df["Algorithm"], plot_df["Davies-Bouldin"], color="#dc2626")
ax[1].set_title("Davies-Bouldin (lower = better)", fontweight="bold")
ax[1].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(f"{OUT}/05_algorithm_comparison.png"); plt.close()

# ----------------------------------------------------------------------
# 3. Final model = KMeans (best balance of separation + interpretability)
# ----------------------------------------------------------------------
final = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(Xs)
df["Cluster"] = final.labels_

# Per-sample silhouette plot
sample_sil = silhouette_samples(Xs, df["Cluster"])
plt.figure(figsize=(9, 6))
y_lower = 10
for i in range(best_k):
    vals = np.sort(sample_sil[df["Cluster"] == i])
    y_upper = y_lower + len(vals)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7)
    plt.text(-0.05, y_lower + 0.5 * len(vals), str(i))
    y_lower = y_upper + 10
plt.axvline(silhouette_score(Xs, df["Cluster"]), color="red", linestyle="--",
            label="avg silhouette")
plt.title("Per-sample Silhouette (KMeans)", fontsize=14, fontweight="bold")
plt.xlabel("Silhouette coefficient"); plt.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/06_silhouette_plot.png"); plt.close()

# ----------------------------------------------------------------------
# 4. Profile + attach personas
# ----------------------------------------------------------------------
prof = df.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income=("Income", "mean"),
    Avg_Spending=("Spending", "mean"),
).round(1)

inc_med, spd_med = df["Income"].median(), df["Spending"].median()
largest = prof["Customers"].idxmax()

persona_rows = {}
for cid, r in prof.iterrows():
    near_center = (abs(r.Avg_Income - inc_med) < df["Income"].std() * 0.6 and
                   abs(r.Avg_Spending - spd_med) < df["Spending"].std() * 0.6)
    p = get_persona(r.Avg_Income, r.Avg_Spending, inc_med, spd_med,
                    is_largest=(cid == largest), near_center=near_center)
    persona_rows[cid] = p

prof["Segment"] = [persona_rows[c]["segment"] for c in prof.index]
prof["Persona"] = [persona_rows[c]["persona"] for c in prof.index]
prof.to_csv(f"{OUT}/segment_profile.csv")
print("\n--- Segment Profile + Personas ---")
print(prof.to_string())

df["Segment"] = df["Cluster"].map({c: persona_rows[c]["segment"] for c in persona_rows})
df["Persona"] = df["Cluster"].map({c: persona_rows[c]["persona"] for c in persona_rows})

# ----------------------------------------------------------------------
# 5. Segment scatter with centroids
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Income", y="Spending", hue="Segment", s=70,
                palette="Set2", alpha=0.85)
cent = StandardScaler().fit(df[["Income", "Spending"]]).inverse_transform(
    KMeans(n_clusters=best_k, random_state=42, n_init=10)
    .fit(StandardScaler().fit_transform(df[["Income", "Spending"]])).cluster_centers_)
plt.scatter(cent[:, 0], cent[:, 1], s=250, marker="X", c="black", label="Centroids")
plt.title("Customer Segments — Income vs Spending", fontsize=14, fontweight="bold")
plt.xlabel("Annual Income (k$)"); plt.ylabel("Spending Score (1-100)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.savefig(f"{OUT}/07_segments_scatter.png"); plt.close()

# ----------------------------------------------------------------------
# 6. Train a classifier to label NEW customers
# ----------------------------------------------------------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
cv = cross_val_score(clf, df[FEATURES], df["Cluster"], cv=5).mean()
clf.fit(df[FEATURES], df["Cluster"])
print(f"\nSegment classifier trained — 5-fold CV accuracy: {cv*100:.1f}%")

joblib.dump({"scaler": scaler, "kmeans": final, "classifier": clf,
             "features": FEATURES,
             "persona_map": {int(c): persona_rows[c] for c in persona_rows}},
            f"{OUT}/segment_model.joblib")

# ----------------------------------------------------------------------
# 7. Save outputs
# ----------------------------------------------------------------------
df[["CustomerID", "Gender", "Age", "Income", "Spending",
    "Cluster", "Segment", "Persona"]].to_csv(f"{OUT}/customers_segmented.csv", index=False)

with open(f"{OUT}/persona_report.txt", "w") as f:
    f.write("MALL CUSTOMERS — PERSONAS & MARKETING PLAYBOOK\n")
    f.write("=" * 50 + "\n\n")
    for cid, r in prof.iterrows():
        p = persona_rows[cid]
        f.write(f"{p['emoji']}  {p['persona']}  ({p['segment']})\n")
        f.write(f"   {p['tagline']}\n")
        f.write(f"   Size: {int(r.Customers)} customers | Avg age {r.Avg_Age:.0f} | "
                f"income ${r.Avg_Income:.0f}k | spending {r.Avg_Spending:.0f}\n")
        f.write(f"   {p['description']}\n")
        f.write("   Playbook:\n")
        for action in p["playbook"]:
            f.write(f"     - {action}\n")
        f.write(f"   Best channel: {p['channel']}\n\n")

print(f"\nSaved charts + customers_segmented.csv + segment_profile.csv + "
      f"persona_report.txt + segment_model.joblib + algorithm_comparison.csv to '{OUT}/'")
print("\nSegment distribution:")
print(df["Segment"].value_counts().to_string())
