import warnings as _w
_w.filterwarnings('ignore')


def predict(product_id: str, target_month: str, panel_df=None) -> dict:
    import pickle
    import numpy as np
    import pandas as pd
    from pathlib import Path

    MODEL_DIR = Path('models')

    with open(MODEL_DIR / 'model_stage1.pkl', 'rb') as f:
        m1 = pickle.load(f)
    with open(MODEL_DIR / 'model_stage2.pkl', 'rb') as f:
        m2 = pickle.load(f)
    with open(MODEL_DIR / 'model_meta.pkl', 'rb') as f:
        meta = pickle.load(f)

    features       = meta['features']
    cat_features   = meta['cat_features']
    zero_threshold = meta['zero_threshold']
    lag_ctx_cols   = meta['lag_context_cols']

    if panel_df is None:
        panel_df = pd.read_parquet(MODEL_DIR / 'panel_features.parquet')
        panel_df['month_dt'] = pd.to_datetime(panel_df['month_dt'])
        panel_df = panel_df.drop_duplicates(subset=['product_id', 'month_dt'])
        for col in cat_features:
            panel_df[col] = panel_df[col].astype('category')

    target_dt = pd.Timestamp(target_month)

    prod_rows = panel_df[panel_df['product_id'] == product_id].sort_values('month_dt')
    if len(prod_rows) == 0:
        return {'error': f'product_id {product_id!r} not found in panel.'}

    is_sparse = bool(prod_rows['is_sparse'].iloc[0])
    panel_end = prod_rows['month_dt'].max()

    predicted_cache = {}

    def get_rev(dt):
        row = prod_rows[prod_rows['month_dt'] == dt]
        if len(row) > 0:
            return float(row['revenue'].iloc[0])
        return predicted_cache.get(dt, np.nan)

    cascade_months = []
    cur = panel_end + pd.DateOffset(months=1)
    while cur <= target_dt:
        cascade_months.append(cur)
        cur = cur + pd.DateOffset(months=1)

    if not cascade_months:
        cascade_months = [target_dt]

    def get_qty(dt, back):
        ref = dt - pd.DateOffset(months=back)
        row = prod_rows[prod_rows['month_dt'] == ref]
        return float(row['quantity'].iloc[0]) if len(row) > 0 else np.nan

    def get_ctx_lag1(dt, col):
        ref = dt - pd.DateOffset(months=1)
        row = prod_rows[prod_rows['month_dt'] == ref]
        return float(row[col].iloc[0]) if len(row) > 0 else np.nan

    static = prod_rows.iloc[-1]

    for month_dt in cascade_months:
        lag1  = get_rev(month_dt - pd.DateOffset(months=1))
        lag2  = get_rev(month_dt - pd.DateOffset(months=2))
        lag3  = get_rev(month_dt - pd.DateOffset(months=3))
        lag12 = get_rev(month_dt - pd.DateOffset(months=12))

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roll3     = np.nanmean([lag1, lag2, lag3])
            qty_roll3 = np.nanmean([get_qty(month_dt, i) for i in [1, 2, 3]])

        row_dict = {
            'revenue_lag12':          lag12,
            'revenue_lag1':           lag1,
            'revenue_lag2':           lag2,
            'revenue_lag3':           lag3,
            'revenue_roll3_mean':     roll3,
            'quantity_lag1':          get_qty(month_dt, 1),
            'quantity_roll3_mean':    qty_roll3,
            'month_num':              month_dt.month,
            'quarter':                (month_dt.month - 1) // 3 + 1,
            'year':                   month_dt.year,
            'is_december':            int(month_dt.month == 12),
            'pack_coefficient':       static['pack_coefficient'],
            'therapeutic_area_clean': static['therapeutic_area_clean'],
            'product_family_clean':   static['product_family_clean'],
        }
        for col in lag_ctx_cols:
            row_dict[f'{col}_lag1'] = get_ctx_lag1(month_dt, col)

        X = pd.DataFrame([row_dict])[features]
        for col in cat_features:
            X[col] = X[col].astype('category')

        prob     = float(m1.predict(X)[0])
        log_pred = float(m2.predict(X)[0])
        rev_pred = float(np.expm1(log_pred))
        final_rev = rev_pred if prob >= zero_threshold else 0.0

        predicted_cache[month_dt] = final_rev
        _last_prob = prob

    final_rev = predicted_cache[target_dt]
    prob      = _last_prob

    months_ahead = (target_dt.year - panel_end.year) * 12 + (target_dt.month - panel_end.month)

    if prob >= zero_threshold and not is_sparse:
        confidence = 'high' if prob > 0.75 and months_ahead <= 3 else 'medium'
    else:
        confidence = 'low'
    if is_sparse:
        confidence = 'low'

    warnings_list = []
    if is_sparse:
        warnings_list.append('Sparse product â€” prediction is low-confidence.')
    if np.isnan(lag12):
        warnings_list.append('revenue_lag12 is NaN (< 12 months history).')
    if target_dt.month == 12:
        warnings_list.append('December forecast â€” procurement freeze may make this unreliable.')
    if months_ahead > 3:
        warnings_list.append(f'{months_ahead} months ahead â€” accuracy degrades beyond 3 months.')

    return {
        'product_id':          product_id,
        'target_month':        target_month,
        'predicted_revenue':   round(final_rev, 2),
        'nonzero_probability': round(prob, 4),
        'confidence':          confidence,
        'is_sparse':           is_sparse,
        'is_december':         (target_dt.month == 12),
        'months_ahead':        months_ahead,
        'warning':             ' | '.join(warnings_list) or None,
    }
