import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.stdout.reconfigure(encoding='utf-8')

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predictor import predict


st.set_page_config(page_title="Pharma Revenue Forecasting", layout="wide")


@st.cache_data
def load_panel_df():
    df = pd.read_parquet("models/panel_features.parquet")
    df["month_dt"] = pd.to_datetime(df["month_dt"])
    return df


@st.cache_data
def load_product_lookup():
    df = pd.read_csv("models/product_lookup.csv")
    if "is_sparse" in df.columns:
        df["is_sparse"] = df["is_sparse"].astype(str).str.lower().isin(["true", "1", "yes"])
    return df


def format_currency(value):
    try:
        return f"{value:,.0f} DZD"
    except Exception:
        return "0 DZD"


def format_probability(value):
    try:
        return f"{value * 100:.1f}%"
    except Exception:
        return "0.0%"


def month_options():
    return list(range(1, 13))


def month_label(month_num):
    return pd.Timestamp(2025, month_num, 1).strftime("%b")


def next_month_defaults():
    today = pd.Timestamp(date.today()).replace(day=1) + pd.DateOffset(months=1)
    year = int(today.year)
    month = int(today.month)
    if year not in [2025, 2026, 2027]:
        year = 2026 if today.year < 2026 else 2027
    return year, month


def build_product_label(row):
    return f"{row['product_id']} - {row['product_name']}"


def confidence_badge(confidence):
    colors = {"high": "#1b8f3a", "medium": "#d97706", "low": "#dc2626"}
    color = colors.get(str(confidence).lower(), "#4b5563")
    return f"<span style='color:{color}; font-weight:700;'>{confidence}</span>"


def monthly_range(start_year, start_month, end_year, end_month):
    start = pd.Timestamp(start_year, start_month, 1)
    end = pd.Timestamp(end_year, end_month, 1)
    if end < start:
        return []
    return list(pd.date_range(start, end, freq="MS"))


def render_history_chart(history_df, forecast_month, forecast_value):
    fig = go.Figure()
    if not history_df.empty:
        fig.add_trace(
            go.Scatter(
                x=history_df["month_dt"],
                y=history_df["revenue"],
                mode="lines+markers",
                name="Historical Revenue",
                line=dict(color="#2563eb", width=3),
                marker=dict(size=6),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[forecast_month],
            y=[forecast_value],
            mode="markers",
            name="Forecast",
            marker=dict(size=14, color="#f97316", symbol="circle"),
        )
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue (DZD)",
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


try:
    panel_df = load_panel_df()
    product_lookup = load_product_lookup()

    if "selected_product" not in st.session_state:
        st.session_state.selected_product = None

    lookup = product_lookup.copy()
    lookup["label"] = lookup.apply(build_product_label, axis=1)
    label_to_row = {row["label"]: row for _, row in lookup.iterrows()}
    labels = lookup["label"].tolist()

    year_default, month_default = next_month_defaults()

    st.sidebar.header("Forecast Controls")
    selected_label = st.sidebar.selectbox(
        "Product",
        labels,
        index=0 if not labels else 0,
    )
    selected_product_row = label_to_row[selected_label]

    st.sidebar.subheader("Forecast Month")
    year_col, month_col = st.sidebar.columns(2)
    selected_year = year_col.selectbox("Year", [2025, 2026, 2027], index=[2025, 2026, 2027].index(year_default) if year_default in [2025, 2026, 2027] else 1)
    selected_month = month_col.selectbox("Month", month_options(), format_func=month_label, index=month_options().index(month_default) if month_default in month_options() else 0)
    forecast_month_dt = pd.Timestamp(selected_year, selected_month, 1)

    forecast_clicked = st.sidebar.button("Forecast", type="primary", use_container_width=True)

    st.sidebar.divider()
    st.sidebar.subheader("Batch Forecast")
    batch_labels = st.sidebar.multiselect("Products", labels)
    from_year_col, from_month_col = st.sidebar.columns(2)
    to_year_col, to_month_col = st.sidebar.columns(2)
    batch_from_year = from_year_col.selectbox("From Year", [2025, 2026, 2027], index=[2025, 2026, 2027].index(year_default) if year_default in [2025, 2026, 2027] else 1, key="batch_from_year")
    batch_from_month = from_month_col.selectbox("From Month", month_options(), format_func=month_label, index=month_options().index(month_default) if month_default in month_options() else 0, key="batch_from_month")
    batch_to_year = to_year_col.selectbox("To Year", [2025, 2026, 2027], index=[2025, 2026, 2027].index(year_default) if year_default in [2025, 2026, 2027] else 1, key="batch_to_year")
    batch_to_month = to_month_col.selectbox("To Month", month_options(), format_func=month_label, index=month_options().index(month_default) if month_default in month_options() else 0, key="batch_to_month")
    batch_clicked = st.sidebar.button("Run Batch", use_container_width=True)

    if forecast_clicked:
        result = predict(selected_product_row["product_id"], str(forecast_month_dt.date()), panel_df=panel_df)
        if "error" in result:
            st.error(result["error"])
        else:
            product_history = panel_df[panel_df["product_id"] == selected_product_row["product_id"]].sort_values("month_dt")
            history_last_24 = product_history.tail(24)
            last_6 = product_history.tail(6).copy()

            st.title(selected_product_row["product_name"])
            st.caption(f"{selected_product_row['therapeutic_area_clean']} | {selected_product_row['product_family_clean']}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Revenue", format_currency(result["predicted_revenue"]))
            c2.metric("Non-Zero Probability", format_probability(result["nonzero_probability"]))
            c3.markdown(f"**Confidence**  \n{confidence_badge(result['confidence'])}", unsafe_allow_html=True)

            if result.get("warning"):
                st.warning(result["warning"])

            render_history_chart(history_last_24, forecast_month_dt, result["predicted_revenue"])

            last_6["Month"] = last_6["month_dt"].dt.strftime("%Y-%m")
            last_6["Revenue"] = last_6["revenue"].map(format_currency)
            st.dataframe(last_6[["Month", "Revenue"]], use_container_width=True, hide_index=True)

    if batch_clicked:
        if not batch_labels:
            st.error("Please select at least one product for batch forecasting.")
        else:
            batch_from = pd.Timestamp(batch_from_year, batch_from_month, 1)
            batch_to = pd.Timestamp(batch_to_year, batch_to_month, 1)
            months = monthly_range(batch_from.year, batch_from.month, batch_to.year, batch_to.month)
            if not months:
                st.error("Batch range is invalid. Please make sure the 'From' month is not after the 'To' month.")
            else:
                rows = []
                product_subset = lookup[lookup["label"].isin(batch_labels)]
                for _, prow in product_subset.iterrows():
                    for month_dt in months:
                        res = predict(prow["product_id"], str(month_dt.date()), panel_df=panel_df)
                        if "error" in res:
                            rows.append({
                                "Product ID": prow["product_id"],
                                "Product Name": prow["product_name"],
                                "Month": month_dt.strftime("%Y-%m"),
                                "Predicted Revenue": None,
                                "Probability": None,
                                "Confidence": None,
                                "Warning": res["error"],
                            })
                        else:
                            rows.append({
                                "Product ID": prow["product_id"],
                                "Product Name": prow["product_name"],
                                "Month": month_dt.strftime("%Y-%m"),
                                "Predicted Revenue": res["predicted_revenue"],
                                "Probability": res["nonzero_probability"],
                                "Confidence": res["confidence"],
                                "Warning": res["warning"],
                            })

                batch_df = pd.DataFrame(rows)
                st.subheader("Batch Results")
                display_df = batch_df.copy()
                display_df["Predicted Revenue"] = display_df["Predicted Revenue"].apply(lambda x: format_currency(x) if pd.notna(x) else "")
                display_df["Probability"] = display_df["Probability"].apply(lambda x: format_probability(x) if pd.notna(x) else "")
                display_df["Warning"] = display_df["Warning"].astype(str).str.replace("—", "-", regex=False)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

                csv_data = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name="batch_forecast_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                agg = batch_df.groupby("Product Name", as_index=False)["Predicted Revenue"].sum()
                agg["Predicted Revenue"] = agg["Predicted Revenue"].fillna(0)
                bar_fig = go.Figure(
                    data=[
                        go.Bar(
                            x=agg["Product Name"],
                            y=agg["Predicted Revenue"],
                            marker_color="#2563eb",
                        )
                    ]
                )
                bar_fig.update_layout(
                    xaxis_title="Product",
                    yaxis_title="Predicted Revenue (DZD)",
                    template="plotly_white",
                    height=420,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(bar_fig, use_container_width=True)

except Exception as exc:
    st.error(f"Application crashed: {exc}")
