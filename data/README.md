# 📥 Dataset — Kaggle "Mall Customer Segmentation Data"

This project uses the **Mall Customer Segmentation** dataset from Kaggle.

## 🔗 Dataset link
**https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python**

## ⬇️ How to download and use it
1. Open the link above (sign in to Kaggle — free).
2. Click **Download**. You get a zip.
3. Unzip it — you'll find **`Mall_Customers.csv`**.
4. Put that file **into this `data/` folder** keeping the name `Mall_Customers.csv`.
5. Re-run the scripts. The loader (`utils.py`) automatically prefers the real
   Kaggle file over the demo file.

```bash
cd project-2-customer-segmentation
python 01_data_cleaning.py
python 02_segmentation.py
streamlit run dashboard.py
```

## 📋 Dataset schema (5 columns, 200 rows)
| Column | Description |
|--------|-------------|
| CustomerID | Unique customer identifier |
| Gender (`Genre` in the original file) | Male / Female |
| Age | Customer age in years |
| Annual Income (k$) | Yearly income in thousands of dollars |
| Spending Score (1-100) | Mall-assigned score of spending behaviour |

> The loader handles the original file's `Genre` column name automatically and
> renames the long columns to `Income` and `Spending` internally.

> A demo file (`sample_demo_data.csv`) with the **same schema** is included so
> the project runs before you download the real one. Replace it for your
> actual submission.
