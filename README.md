# Fraud Detection Using Machine Learning

Fraud detection on a synthetic mobile-money transaction dataset (AIML Dataset), combining exploratory data analysis, a scikit-learn classification pipeline, and a Streamlit prediction app.

## Dataset

[AIML Fraud Detection Dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset) (Kaggle) — ~6.4M transactions with a fraud rate of about 0.1%.

The dataset is **not** committed to this repo (493 MB, over GitHub's 100 MB file limit). Download `AIML Dataset.csv` from the link above and place it in the repo root to re-run the notebook.

## Contents

| File | Description |
| --- | --- |
| `analysis_model.ipynb` | EDA (transaction types, fraud by type, amount distributions, balance-difference features, correlation heatmap) and model training/evaluation |
| `fraud_detection_pipeline.pkl` | Trained scikit-learn pipeline (joblib) |
| `fraud_dectection.py` | Streamlit app for interactive predictions |
| `Fraud Detection Using Machine Learning.txt` | Reference links (video tutorial and dataset) |

## Model

- **Features:** transaction type, amount, sender/receiver balances before and after; engineered balance-difference features explored in the notebook
- **Pipeline:** `ColumnTransformer` (StandardScaler on numeric, OneHotEncoder on transaction type) into `LogisticRegression(class_weight="balanced", max_iter=1000)`
- **Split:** stratified 70/30 train/test
- **Results:** 94.5% accuracy; fraud class recall 0.94, precision 0.02 (class weighting trades precision for recall, which is the right trade-off for fraud screening)

## Run the app

```bash
pip install streamlit pandas joblib scikit-learn
streamlit run fraud_dectection.py
```
