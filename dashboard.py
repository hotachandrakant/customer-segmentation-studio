"""
dashboard.py
------------
Advanced + creative customer-segmentation app (Streamlit + Plotly).

4 tabs:
  🔮 Segment     - pick algorithm + k, see clusters live (2D & 3D)
  📊 Profiles    - radar chart + profile table + segment sizes
  🧬 Personas    - auto-generated persona cards with marketing playbooks
  🎯 Classify Me - enter a new customer's details -> predict their segment

Run:
    pip install -r requirements.txt
    python 02_segmentation.py     # once, to create the saved model (optional)
    streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from utils import load_customers
from personas import get_persona

st.set_page_config(page_title="Customer Segmentation Studio", layout="wide", page_icon="🧬")


@st.cache_data
def get_data():
    return load_customers()


df, source = get_data()
FEATURES = ["Age", "Income", "Spending"]

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")
algo_name = st.sidebar.selectbox("Algorithm",
                                 ["KMeans", "Agglomerative", "GaussianMixture"])
k = st.sidebar.slider("Number of segments (k)", 2, 10, 5)
feats = st.sidebar.multiselect("Cluster on", FEATURES, default=["Income", "Spending"])
st.sidebar.caption(f"Data source: {source}")

if len(feats) < 2:
    st.warning("Pick at least 2 features."); st.stop()

Xs = StandardScaler().fit_transform(df[feats])
if algo_name == "KMeans":
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
elif algo_name == "Agglomerative":
    model = AgglomerativeClustering(n_clusters=k)
else:
    model = GaussianMixture(n_components=k, random_state=42)
labels = model.fit_predict(Xs)
df["Cluster"] = labels.astype(str)
sil = silhouette_score(Xs, labels)

# Name segments via personas (based on income/spending averages)
inc_med, spd_med = df["Income"].median(), df["Spending"].median()
prof = df.groupby("Cluster").agg(Avg_Income=("Income", "mean"),
                                 Avg_Spending=("Spending", "mean"),
                                 Customers=("CustomerID", "count"))
largest = prof["Customers"].idxmax()
seg_map, persona_map = {}, {}
for cid, r in prof.iterrows():
    near = (abs(r.Avg_Income - inc_med) < df["Income"].std() * 0.6 and
            abs(r.Avg_Spending - spd_med) < df["Spending"].std() * 0.6)
    p = get_persona(r.Avg_Income, r.Avg_Spending, inc_med, spd_med,
                    is_largest=(cid == largest), near_center=near)
    seg_map[cid] = p["segment"]; persona_map[p["segment"]] = p
df["Segment"] = df["Cluster"].map(seg_map)

st.title("🧬 Customer Segmentation Studio")
st.caption("Thiranex Internship — Project 2 (Advanced) | Data Analyst")

c = st.columns(4)
c[0].metric("Customers", len(df))
c[1].metric("Algorithm", algo_name)
c[2].metric("Segments", k)
c[3].metric("Silhouette", f"{sil:.3f}")

t1, t2, t3, t4 = st.tabs(["🔮 Segment", "📊 Profiles", "🧬 Personas", "🎯 Classify Me"])

# ===== Segment =====
with t1:
    col1, col2 = st.columns(2)
    with col1:
        xcol = "Income" if "Income" in feats else feats[0]
        ycol = "Spending" if "Spending" in feats else feats[1]
        st.plotly_chart(px.scatter(df, x=xcol, y=ycol, color="Segment",
                                   hover_data=["Age", "Gender"],
                                   title=f"{xcol} vs {ycol}",
                                   color_discrete_sequence=px.colors.qualitative.Set2),
                        use_container_width=True)
    with col2:
        st.plotly_chart(px.scatter_3d(df, x="Age", y="Income", z="Spending",
                                      color="Segment", title="3D view",
                                      color_discrete_sequence=px.colors.qualitative.Set2),
                        use_container_width=True)

# ===== Profiles =====
with t2:
    metrics = ["Age", "Income", "Spending"]
    norm = pd.DataFrame(MinMaxScaler().fit_transform(df[metrics]), columns=metrics)
    norm["Segment"] = df["Segment"].values
    radar = norm.groupby("Segment")[metrics].mean()
    fig = go.Figure()
    for seg in radar.index:
        fig.add_trace(go.Scatterpolar(r=radar.loc[seg].tolist() + [radar.loc[seg][0]],
                                      theta=metrics + [metrics[0]], fill="toself", name=seg))
    fig.update_layout(title="Segment Radar (normalised)",
                      polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    col1, col2 = st.columns((3, 2))
    col1.plotly_chart(fig, use_container_width=True)
    table = df.groupby("Segment").agg(Customers=("CustomerID", "count"),
                                      Avg_Age=("Age", "mean"),
                                      Avg_Income=("Income", "mean"),
                                      Avg_Spending=("Spending", "mean")).round(1)
    col2.dataframe(table, use_container_width=True)

# ===== Personas =====
with t3:
    cols = st.columns(2)
    for i, seg in enumerate(sorted(df["Segment"].unique())):
        p = persona_map[seg]
        sz = (df["Segment"] == seg).sum()
        pr = df[df["Segment"] == seg][["Age", "Income", "Spending"]].mean()
        with cols[i % 2]:
            st.markdown(f"### {p['emoji']} {p['persona']}")
            st.caption(f"{seg} — {p['tagline']}")
            st.write(f"👥 **{sz}** customers · 🎂 Age {pr['Age']:.0f} · "
                     f"💰 ${pr['Income']:.0f}k · 🛒 Spend {pr['Spending']:.0f}")
            st.write(p["description"])
            st.write("**Playbook:** " + "; ".join(p["playbook"]))
            st.info(f"📣 Best channel: {p['channel']}")
            st.divider()

# ===== Classify Me =====
with t4:
    st.write("Enter a new customer's details and instantly see which segment "
             "they belong to (trained on the current clustering).")
    cc = st.columns(3)
    age = cc[0].number_input("Age", 18, 90, 30)
    income = cc[1].number_input("Annual Income (k$)", 10, 200, 60)
    spend = cc[2].number_input("Spending Score (1-100)", 1, 100, 70)
    if st.button("🎯 Predict segment"):
        # train a quick classifier on the current labels
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=150, random_state=42)
        clf.fit(df[FEATURES], df["Segment"])
        pred = clf.predict([[age, income, spend]])[0]
        p = persona_map.get(pred)
        st.success(f"This customer belongs to **{pred}**")
        if p:
            st.markdown(f"### {p['emoji']} {p['persona']}")
            st.write(p["description"])
            st.write("**Playbook:** " + "; ".join(p["playbook"]))
