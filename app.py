import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from prophet import Prophet

st.set_page_config(page_title="Coffee Sales Forecaster", layout="wide")
st.title("☕ Coffee Sales Forecaster")

st.markdown("""
Upload your `Coffe_sales.csv` **or** provide a GitHub raw URL.  
The app aggregates to **daily revenue** and fits a **Prophet** model (no saved models).
""")

default_url = "https://raw.githubusercontent.com/RahulBhattacharya1/ai_coffee_forecast/refs/heads/main/data/coffee_sales.csv"
raw_url = st.text_input("GitHub Raw CSV URL (optional if you upload)", value=default_url)

uploaded = st.file_uploader("Upload Coffe_sales.csv", type=["csv"])

horizon = st.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30, step=1)
use_weekly = st.checkbox("Weekly seasonality", value=True)
use_daily  = st.checkbox("Daily seasonality", value=True)

@st.cache_data(show_spinner=False)
def load_csv(raw_url, uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    try:
        return pd.read_csv(raw_url)
    except Exception as e:
        raise RuntimeError(f"Could not load CSV. Check URL or upload a file. Error: {e}")

def to_daily(df):
    # Expecting columns Date, money
    if 'Date' not in df.columns or 'money' not in df.columns:
        raise ValueError("CSV must include columns 'Date' and 'money'.")
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['money'] = pd.to_numeric(df['money'], errors='coerce').fillna(0)
    daily = (
        df.groupby('Date', as_index=False)['money']
          .sum()
          .rename(columns={'Date':'ds','money':'y'})
          .sort_values('ds')
    )
    return daily

def fit_prophet(daily, horizon, weekly=True, daily_seas=True):
    m = Prophet(
        weekly_seasonality=weekly,
        daily_seasonality=daily_seas,
        yearly_seasonality=False,
        seasonality_mode='additive'
    )
    m.fit(daily)
    future = m.make_future_dataframe(periods=horizon, freq='D')
    fcst = m.predict(future)
    return m, fcst

try:
    df_raw = load_csv(raw_url, uploaded)
    st.success(f"Loaded dataset with shape {df_raw.shape}")
    with st.expander("Preview data"):
        st.dataframe(df_raw.head(20), use_container_width=True)

    daily = to_daily(df_raw)
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.line(daily, x='ds', y='y', title='Daily Revenue (Actual)')
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        st.metric("Total Revenue", f"{daily['y'].sum():,.2f}")
        st.metric("Avg Daily Revenue", f"{daily['y'].mean():,.2f}")
        st.metric("Days of Data", f"{daily.shape[0]}")

    # Simple split: last 14 days as validation if available
    h = min(14, max(1, daily.shape[0] // 5))
    train = daily.iloc[:-h] if daily.shape[0] > h else daily.copy()
    test  = daily.iloc[-h:] if daily.shape[0] > h else pd.DataFrame(columns=['ds','y'])

    with st.spinner("Training Prophet..."):
        m, fcst = fit_prophet(train, horizon, weekly=use_weekly, daily_seas=use_daily)

    # Join forecasts with actuals
    plot_df = fcst[['ds','yhat','yhat_lower','yhat_upper']].merge(daily, on='ds', how='left')
    fig_fc = px.line(plot_df, x='ds', y=['y','yhat'], title='Actual vs Forecast')
    st.plotly_chart(fig_fc, use_container_width=True)

    # Show forecast table for future only
    future_only = plot_df[plot_df['ds'] > daily['ds'].max()].copy()
    future_only = future_only[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={
        'ds':'Date','yhat':'Forecast','yhat_lower':'Lo','yhat_upper':'Hi'
    })
    st.subheader("Forecast (Future Days)")
    st.dataframe(future_only, use_container_width=True)

    with st.expander("Trend and Seasonality"):
        st.write("Trend / weekly / daily components (static images from Prophet).")
        st.pyplot(m.plot(fcst))
        st.pyplot(m.plot_components(fcst))

except Exception as e:
    st.error(str(e))
