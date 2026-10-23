
# Pharma Revenue Forecasting
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://ai-powered-phishing-detector.streamlit.app/](https://medis-revenue-forecasting.streamlit.app/))

Monthly revenue forecasting **per product** for a Tunisian pharmaceutical group (MEDIS & NEAPOLIS), with an interactive Streamlit app.

A two-stage global LightGBM model (zero classifier + log-revenue regressor) trained on ~290k cleaned invoice lines (Jan 2023 - Apr 2026), served through a web app that forecasts up to 12 months ahead for single products or batches.

> **Status: first iteration / portfolio project.** The pipeline and app work end to end, but forecast accuracy is limited (see [Results](#results) and [Limitations](#limitations)). I document the weaknesses hoping I can come back and investigate it 

## Pipeline

| Step | Notebook | What it does |
| ---- | -------- | ------------ |
| 1 | `01_exploration.ipynb` | Structure, missing values, duplicates and odd values in the three raw tables |
| 2 | `02_cleaning.ipynb` | Cleaning and merge: duplicates, returns, zero-price lines, category standardisation |
| 3 | `03_aggregation_v2.ipynb` | Builds the `product x month` panel, client-mix features, calendar and lag features |
| 4 | `04_eda.ipynb` | EDA, with every finding ending in a modelling decision |
| 5 | `05_forecasting_v2.ipynb` | Training, evaluation, export of model artefacts |
| - | `predictor.py`, `app.py` | Recursive forecaster and Streamlit UI |

**Data:** star schema of 3 tables (sales facts, products, clients). Cleaning takes 295,338 raw lines down to 290,542 sales lines (2,518 duplicates, 1,525 returns and 753 zero-price lines removed).

### Key design decisions

- **Grain: product x month.** Compared with product x client x month it roughly halves the zero rate and skew and doubles the lag-1 correlation (0.26 -> 0.57). Client information is kept as aggregated features (number of clients, export share, hospital share, ...).
- **Zero-inflated, skewed target** (~30% zero months among non-sparse products) -> two stages: a classifier for "any sales this month?", then a regressor on `log1p(revenue)` using only active months.
- **Strongest signal: same month last year** (`revenue_lag12`, r ~ 0.72 on the log scale). It captures seasonality, including the pronounced December dip (December avg ~ 39% of January).
- **No leakage:** same-month context variables are replaced by their lag-1 version; same-month quantity and price columns are dropped.
- **Strict time split:** train up to Jul 2025, validation Aug-Nov 2025 (early stopping, threshold tuning), test Dec 2025-Apr 2026.
- **Why not deep learning:** at most 40 monthly points per series, and about a quarter of products have fewer than 3 active months.

## The app

- **Single forecast:** pick a product and a month -> predicted revenue, probability of non-zero sales, confidence level, warnings, and a 24-month history chart.
- **Batch forecast:** several products x a month range, with a results table, CSV download and a chart.
- **Horizons beyond the next month** use a recursive strategy: each month's forecast feeds the lags of the next one.
- **Confidence** is a simple rule, not a statistical interval:
  - *high*: non-sparse product, sale judged likely (p > 0.75), at most 3 months after the last data month
  - *medium*: sale judged likely but further out or less certain
  - *low*: sparse product or sale judged unlikely

## Results

Evaluated on Dec 2025-Apr 2026, 349 products, 1,562 product-months. **These are one-step-ahead results (true lags). The app's multi-month recursive forecasts have not been back-tested and should be expected to be worse.**

| Metric | Value |
| ------------------------------------- | ------------------------ |
| Median absolute % error | 48.1% |
| WAPE (revenue-weighted error) | 59.3% |
| Mean MAPE (incl. / excl. December) | 131.4% / 139.0% |
| Total predicted vs. actual revenue | 79.5 M vs. 136.4 M (58%) |
| Products with MAPE < 50% | 37% |
| Top-10% revenue products: MAPE / WAPE | 94% / 54% |

By therapeutic area (MAPE): CNS & Sexology 36%, Others 62%, Cardio/Metabolic/Respiratory 78%, Food Supplements 115%, Gastro/Trauma/Infectiology 134%, Rea/Radio/Dental 157%, Oncology & Immunology 412%.

The error is much lower on large, steady products than on the long tail of lumpy, intermittently ordered ones (mostly injectables and oncology).

## Limitations

- **Systematic under-forecasting** (58% of actual revenue). Likely causes, still to be verified: the hard zero gate (198 active months were forecast as zero), the bias from back-transforming a log-scale model, and large infrequent orders.
- **Multi-step forecasts are not back-tested**; confidence levels are heuristic and no prediction intervals are provided.
- **Feature mismatch at long horizons:** quantity and context lags are missing beyond the first forecast month, whereas the model was trained on complete rows.
- **Error propagation:** a zero forecast feeds zeros into later months' lags.
- **Training cutoff:** final models were fitted up to Jul 2025 and not refitted on the most recent months.
- **Short history** (<= 40 months) and many sparse products.

## Roadmap

- [ ] Compare against baselines (naive, seasonal naive, rolling mean) and report them
- [ ] Reduce the bias: smearing correction, expected-value forecast instead of a hard gate, Tweedie objective
- [ ] Rolling-origin back-test of the recursive forecaster at horizons 1-12 months
- [ ] Prediction intervals (quantile regression / conformal) to replace the heuristic confidence
- [ ] Refit on all data before forecasting
- [ ] Separate models/evaluation for hospital- and export-driven products
- [ ] Confirm the currency label in the app (currently shown as DZD; the company is Tunisian)


## Repository layout

```
.
|-- app.py                  # Streamlit UI
|-- predictor.py             # recursive two-stage forecasting
|-- requirements.txt
|-- notebooks/              # 01 -> 05, run in order
|-- data/                   # raw and processed 
`-- models/                 # generated
```

## Tech stack

Python, pandas, NumPy, LightGBM, scikit-learn, Streamlit, Plotly.
