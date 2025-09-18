import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from prophet import Prophet

st.set_page_config(page_title="Coffee Sales Forecaster", layout="wide")
st.title("Coffee Sales Forecaster")

st.markdown(
    "Upload your CSV or provide a GitHub raw URL. "
    "The app aggregates to daily revenue and fits a Prophet model (no saved models)."
)

# Default to your repo file
DEFAULT_RAW_URL = (
    "https://raw.githubusercontent.com/RahulBhattacharya1/ai_coffee_forecast/"
    "refs/heads/main/data/coffee_sales.csv"
)

raw_url = st.text_input("GitHub Raw CSV URL (optional if you upload)", value=DEFAULT_RAW_URL)
uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])

# Forecast controls
horizon = st.slider("Forecast Horizon (days)", min_value=7, max_value=120, value=30, step=1)
use_weekly = st.checkbox("Weekly seasonality", value=True)
use_daily = st.checkbox("Daily seasonality", value=True)
visible_past_days = st.number_input(
    "History window to show on chart (days)",
    min_value=30,
    max_value=365,
    value=90,
    step=15
)

@st.cache_data(show_spinner=False)
def load_csv(raw_url: str, uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if not raw_url:
        raise RuntimeError("No URL or file provided.")
    return pd.read_csv(raw_url)

def pick_column(candidates, cols_lower):
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    return None

def to_daily(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # map lowercase -> original name
    cols_lower = {c.lower(): c for c in df.columns}

    def pick_column(candidates):
        for c in candidates:
            if c in cols_lower:
                return cols_lower[c]
        return None

    date_col = pick_column(["date", "order_date", "ds", "timestamp", "datetime"])
    value_col = pick_column(["money", "amount", "revenue", "sales", "total", "price", "y"])

    if date_col is None or value_col is None:
        raise ValueError(
            "CSV must include a date column and a numeric revenue column. "
            "Accepted date names: Date/Order_Date/ds/Timestamp/Datetime; "
            "value names: money/amount/revenue/sales/total/price/y."
        )

    # coerce types
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df[value_col] = df[value_col].fillna(0)

    # group by day using Grouper to ensure a named column
    daily = (
        df.groupby(pd.Grouper(key=date_col, freq="D"))[value_col]
          .sum()
          .reset_index()
    )

    # force canonical names regardless of whatever pandas called them
    daily.columns = ["ds", "y"]
    daily = daily.sort_values("ds")
    return daily

def fit_prophet(daily_df: pd.DataFrame, horizon_days: int, weekly=True, daily_seas=True):
    m = Prophet(
        weekly_seasonality=weekly,
        daily_seasonality=daily_seas,
        yearly_seasonality=False,
        seasonality_mode="additive",
    )
    m.fit(daily_df)
    future = m.make_future_dataframe(periods=horizon_days, freq="D")
    fcst = m.predict(future)
    return m, fcst

try:
    df_raw = load_csv(raw_url, uploaded)
    st.success(f"Loaded dataset with shape {df_raw.shape}")
    with st.expander("Preview data"):
        st.dataframe(df_raw.head(30), use_container_width=True)

    daily = to_daily(df_raw)

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.line(daily, x="ds", y="y", title="Daily Revenue (Actual)")
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        st.metric("Total Revenue", f"{daily['y'].sum():,.2f}")
        st.metric("Avg Daily Revenue", f"{daily['y'].mean():,.2f}")
        st.metric("Days of Data", f"{daily.shape[0]}")

    # Simple validation split: last ~20% capped between 7 and 28 days
    h_val = min(28, max(7, daily.shape[0] // 5))
    if daily.shape[0] > h_val:
        train = daily.iloc[:-h_val].copy()
        test = daily.iloc[-h_val:].copy()
    else:
        train = daily.copy()
        test = pd.DataFrame(columns=["ds", "y"])

    with st.spinner("Training Prophet..."):
        m, fcst = fit_prophet(train, horizon, weekly=use_weekly, daily_seas=use_daily)

    # Join forecasts with actuals
    plot_df = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(daily, on="ds", how="left")

    # Focused chart: last N past days + all future
    max_actual_date = daily["ds"].max()
    cutoff = max_actual_date - pd.Timedelta(days=int(visible_past_days))
    plot_recent = plot_df[plot_df["ds"] >= cutoff].copy()

    # Future table
    future_only = plot_df[plot_df["ds"] > max_actual_date].copy()
    future_only = future_only[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
        columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lo", "yhat_upper": "Hi"}
    )
    last_forecast_date = plot_df["ds"].max()
    future_days = future_only.shape[0]

    st.subheader("Forecast View")
    st.caption(
        f"Showing last {visible_past_days} days of history + {future_days} future days "
        f"(ends {last_forecast_date.date()})."
    )

    fig_fc = px.line(
        plot_recent,
        x="ds",
        y=["y", "yhat"],
        title="Actual vs Forecast (focused view)",
        labels={"ds": "Date", "value": "Revenue"},
    )
    fig_fc.update_layout(legend_title_text="")
    st.plotly_chart(fig_fc, use_container_width=True)

    st.subheader("Forecast (Future Days)")
    st.dataframe(future_only, use_container_width=True)
    st.caption(
        f"Forecast horizon = {future_days} days (slider). "
        f"Last forecast date = {last_forecast_date.date()}."
    )

    with st.expander("Trend and Seasonality"):
        st.pyplot(m.plot(fcst))
        st.pyplot(m.plot_components(fcst))

except Exception as e:
    st.error(str(e))
