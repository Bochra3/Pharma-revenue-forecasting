# Model artifacts

The Streamlit app expects these locally generated or authorized files in this folder:

- `panel_features.parquet`
- `product_lookup.csv`
- `model_stage1.pkl`
- `model_stage2.pkl`
- `model_meta.pkl`

These generated model and feature files are tracked in Git so the dashboard can run in deployments and fresh clones without a separate model generation step.
