# BridgeCompliance Advisory — Market Validation Dashboard

Cross-border legal and regulatory advisory analytics for Dubai (DIFC/ADGM) and Singapore (MAS).

## Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to share.streamlit.io → New app
3. Set **Main file path**: `app.py`
4. Deploy

## Usage

Upload `BridgeCompliance_Preprocessed_Dataset.xlsx` in the sidebar when the app loads.

## Tabs

| Tab | Method | Business Question |
|---|---|---|
| 1 — Market Comparison | Descriptive statistics | Which market has stronger demand? |
| 2 — WTP Drivers | Spearman correlation | What drives willingness to pay? |
| 3 — Predicting Engagement | Decision Tree, RF, GBM | Who will engage? |
| 4 — Predicting Revenue | Linear, Ridge, Lasso regression | How much will they pay? |
| 5 — Segments & Bundles | K-Prototypes + Apriori | How to segment and bundle services? |
| 6 — Recommendations | Written synthesis | Strategic launch decisions |
