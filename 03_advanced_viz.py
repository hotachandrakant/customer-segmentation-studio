"""
03_advanced_viz.py
------------------
Creative + advanced visualisations on top of the segmentation:
  1. 3D scatter (Age x Income x Spending) coloured by segment
  2. t-SNE non-linear 2D embedding of the segments
  3. Hierarchical clustering DENDROGRAM (scipy)
  4. RADAR / spider chart comparing each segment's profile
  5. Persona "cards" rendered as an HTML page you can open in a browser

Run AFTER 02_segmentation.py (it reads outputs/customers_segmented.csv).

Usage:
    python 03_advanced_viz.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3d projection)
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from personas import ARCHETYPES, STANDARD

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(f"{OUT}/customers_segmented.csv")
segments = sorted(df["Segment"].unique())
palette = dict(zip(segments, sns.color_palette("Set2", len(segments))))

# ----------------------------------------------------------------------
# 1. 3D scatter
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
for seg in segments:
    s = df[df["Segment"] == seg]
    ax.scatter(s["Age"], s["Income"], s["Spending"], label=seg,
               color=palette[seg], s=40, alpha=0.8)
ax.set_xlabel("Age"); ax.set_ylabel("Income (k$)"); ax.set_zlabel("Spending")
ax.set_title("Customer Segments in 3D", fontsize=14, fontweight="bold")
ax.legend(bbox_to_anchor=(1.1, 1), loc="upper left", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/08_segments_3d.png"); plt.close()

# ----------------------------------------------------------------------
# 2. t-SNE embedding
# ----------------------------------------------------------------------
Xs = StandardScaler().fit_transform(df[["Age", "Income", "Spending"]])
emb = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(Xs)
df["tsne1"], df["tsne2"] = emb[:, 0], emb[:, 1]
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="tsne1", y="tsne2", hue="Segment", palette=palette, s=70)
plt.title("t-SNE Embedding of Customer Segments", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/09_tsne.png"); plt.close()

# ----------------------------------------------------------------------
# 3. Dendrogram
# ----------------------------------------------------------------------
Z = linkage(Xs, method="ward")
plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode="lastp", p=25, leaf_rotation=90, leaf_font_size=9)
plt.title("Hierarchical Clustering Dendrogram (Ward linkage)",
          fontsize=14, fontweight="bold")
plt.xlabel("Customers (truncated)"); plt.ylabel("Distance")
plt.tight_layout(); plt.savefig(f"{OUT}/10_dendrogram.png"); plt.close()

# ----------------------------------------------------------------------
# 4. Radar / spider chart per segment
# ----------------------------------------------------------------------
metrics = ["Age", "Income", "Spending"]
norm = pd.DataFrame(MinMaxScaler().fit_transform(df[metrics]), columns=metrics)
norm["Segment"] = df["Segment"].values
radar = norm.groupby("Segment")[metrics].mean()

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for seg in radar.index:
    vals = radar.loc[seg].tolist(); vals += vals[:1]
    ax.plot(angles, vals, label=seg, color=palette[seg], linewidth=2)
    ax.fill(angles, vals, color=palette[seg], alpha=0.1)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics)
ax.set_title("Segment Profiles (radar, normalised)", fontsize=14,
             fontweight="bold", pad=20)
ax.legend(bbox_to_anchor=(1.25, 1.1), fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/11_radar_profiles.png"); plt.close()

# ----------------------------------------------------------------------
# 5. Persona cards -> HTML
# ----------------------------------------------------------------------
templates = list(ARCHETYPES.values()) + [STANDARD]
tpl_by_segment = {t["segment"]: t for t in templates}

sizes = df["Segment"].value_counts().to_dict()
profile = df.groupby("Segment")[["Age", "Income", "Spending"]].mean().round(0)

cards = []
for seg in segments:
    t = tpl_by_segment.get(seg, STANDARD)
    p = profile.loc[seg]
    actions = "".join(f"<li>{a}</li>" for a in t["playbook"])
    cards.append(f"""
    <div class="card">
      <div class="emoji">{t['emoji']}</div>
      <h2>{t['persona']}</h2>
      <div class="seg">{seg}</div>
      <p class="tag">{t['tagline']}</p>
      <div class="stats">
        <span>👥 {sizes.get(seg,0)} customers</span>
        <span>🎂 Age {p['Age']:.0f}</span>
        <span>💰 ${p['Income']:.0f}k</span>
        <span>🛒 Spend {p['Spending']:.0f}</span>
      </div>
      <p>{t['description']}</p>
      <strong>Marketing playbook</strong>
      <ul>{actions}</ul>
      <div class="channel">📣 {t['channel']}</div>
    </div>""")

html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Customer Personas</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         background:#0f172a; color:#e2e8f0; padding:30px; }}
  h1 {{ text-align:center; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
          gap:20px; max-width:1200px; margin:0 auto; }}
  .card {{ background:#1e293b; border-radius:16px; padding:22px;
          box-shadow:0 6px 18px rgba(0,0,0,.35); border:1px solid #334155; }}
  .emoji {{ font-size:42px; }}
  .card h2 {{ margin:6px 0 2px; color:#38bdf8; }}
  .seg {{ font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; }}
  .tag {{ font-style:italic; color:#cbd5e1; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:10px; margin:10px 0; font-size:13px; }}
  .stats span {{ background:#0f172a; padding:4px 10px; border-radius:20px; }}
  ul {{ margin:6px 0 12px 18px; }} li {{ margin:3px 0; }}
  .channel {{ background:#164e63; padding:8px 12px; border-radius:10px;
             font-size:13px; display:inline-block; }}
</style></head><body>
<h1>🧬 Customer Personas</h1>
<div class="grid">{''.join(cards)}</div>
</body></html>"""

with open(f"{OUT}/persona_cards.html", "w") as f:
    f.write(html)

print("Saved advanced visuals to outputs/:")
print("  08_segments_3d.png, 09_tsne.png, 10_dendrogram.png,")
print("  11_radar_profiles.png, persona_cards.html (open in a browser)")
