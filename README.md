# 🧬  Customer Segmentation Studio

**Thiranex Internship — Data Analyst Track**

Segment mall customers, validate the clustering with multiple algorithms, build
vivid marketing **personas**, and ship a model that can classify brand-new
customers — all wrapped in an interactive 4-tab dashboard.

> 🔗 **Dataset:** https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
> (download `Mall_Customers.csv` into `data/` — see `data/README.md`)

---

## ⭐ What makes this advanced + creative
**Advanced ML**
- 4 clustering algorithms compared — **KMeans, Agglomerative, Gaussian Mixture, DBSCAN**
- 3 validation metrics — **Silhouette, Davies-Bouldin, Calinski-Harabasz**
- **Elbow + Silhouette** to choose k, plus a **per-sample silhouette plot**
- **Random-Forest classifier** trained on the segments (≈97% CV accuracy) and
  saved with joblib, so any new customer can be auto-assigned a segment

**Creative analysis & visuals**
- **3D scatter** across Age × Income × Spending
- **t-SNE** non-linear embedding of the segments
- **Hierarchical dendrogram** (Ward linkage)
- **Radar / spider charts** profiling each segment
- **Persona cards** — every segment becomes a named character (e.g. *Priya the
  Power Shopper* 💎, *Aisha the Aspirational Spender* 🛍️) with a marketing
  playbook and best channel — rendered as `outputs/persona_cards.html`

**Interactive dashboard (4 tabs)**
1. 🔮 **Segment** — switch algorithm + k, live 2D and 3D cluster views
2. 📊 **Profiles** — interactive radar chart + profile table
3. 🧬 **Personas** — persona cards with playbooks
4. 🎯 **Classify Me** — type a new customer's Age/Income/Spending → predicted segment

## 🧩 Personas discovered (demo run, k = 5)
| Persona | Segment | Behaviour |
|---------|---------|-----------|
| 💎 Priya the Power Shopper | Premium / Target | High income + high spending |
| 🧐 Rohan the Reluctant Earner | Careful | High income, low spending (untapped) |
| 🛍️ Aisha the Aspirational Spender | Careless | Low income, high spending |
| 🪙 Sanjay the Sensible Saver | Sensible / Frugal | Low income, low spending |
| ⚖️ Maya the Mainstream Middle | Standard | The core majority |

## 🗂 File structure
```
project-2-customer-segmentation/
├── README.md
├── requirements.txt
├── utils.py                  # robust loader (Genre->Gender, missing values)
├── personas.py               # persona archetypes + marketing playbooks
├── make_demo_data.py         # creates a schema-matching demo CSV
├── 01_data_cleaning.py       # quality report -> outputs/customers_clean.csv
├── 02_segmentation.py        # multi-algo compare + validation + classifier
├── 03_advanced_viz.py        # 3D, t-SNE, dendrogram, radar, persona cards
├── dashboard.py              # 4-tab interactive Streamlit app
├── data/
│   ├── README.md             # Kaggle link + schema
│   └── sample_demo_data.csv  # demo data (replace with real Mall_Customers.csv)
└── outputs/
    ├── 01_distributions.png ... 11_radar_profiles.png   (charts)
    ├── algorithm_comparison.csv
    ├── customers_clean.csv
    ├── customers_segmented.csv
    ├── segment_profile.csv
    ├── persona_report.txt
    ├── persona_cards.html       # open in a browser
    └── segment_model.joblib     # saved scaler + kmeans + classifier
```

## ▶️ How to run
```bash
# 1. install
pip install -r requirements.txt

# 2. add data (real Kaggle file recommended) OR generate demo:
python make_demo_data.py

# 3. run the pipeline
python 01_data_cleaning.py     # clean + quality report
python 02_segmentation.py      # compare algorithms, label, train classifier
python 03_advanced_viz.py      # 3D / t-SNE / dendrogram / radar / persona cards

# 4. launch the dashboard
streamlit run dashboard.py
```

## 🛠 Tech stack
Python · pandas · NumPy · scikit-learn (KMeans, Agglomerative, GMM, DBSCAN, PCA,
t-SNE, RandomForest, silhouette/DB/CH metrics) · SciPy (dendrogram) · joblib ·
Matplotlib · Seaborn · Plotly · Streamlit

## 📚 What I learned
Comparing clustering algorithms with real validation metrics, non-linear
embeddings (t-SNE), hierarchical clustering, persona-driven storytelling, and
deploying a classifier to score new customers.
