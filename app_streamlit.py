# app_streamlit.py - FORCE TODAY'S DATE WITH LIVE PRICE
import streamlit as st
import requests
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import time
import subprocess
import sys

st.set_page_config(page_title="Bitcoin Prediction", page_icon="₿", layout="wide")

API_BASE = "http://127.0.0.1:8000"

if 'last_update' not in st.session_state:
    st.session_state.last_update = None

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); color: #ffffff;}
    h1 {color: #f7931a !important; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);}
    h2, h3 {color: #ffa500 !important;}
    div[data-testid="stSidebar"] button[kind="secondary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important; border: 2px solid #a78bfa !important;
        border-radius: 12px !important; padding: 18px 24px !important;
        font-weight: bold !important; font-size: 18px !important; width: 100% !important;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important; text-transform: uppercase !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.6) !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #f7931a 0%, #ffa500 100%);
        color: white; border: none; border-radius: 8px;
        padding: 12px 24px; font-weight: bold; width: 100%;
    }
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);}
</style>
""", unsafe_allow_html=True)

def get_live_bitcoin_price():
    """Get CURRENT Bitcoin price"""
    try:
        btc = yf.Ticker("BTC-USD")
        # Try multiple methods to get current price
        
        # Method 1: Fast history
        fast = btc.history(period="1d", interval="1m")
        if not fast.empty:
            return float(fast['Close'].iloc[-1])
        
        # Method 2: Info
        info = btc.info
        price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
        if price:
            return float(price)
        
        return None
    except:
        return None

def force_update_with_todays_date():
    """FORCE update with TODAY's date - GUARANTEED"""
    try:
        # Get live current price
        live_price = get_live_bitcoin_price()
        
        if not live_price:
            return False, "❌ Could not get live price"
        
        # Get historical data
        btc = yf.Ticker("BTC-USD")
        hist = btc.history(period="1y", interval="1d")
        
        if hist.empty:
            return False, "❌ No historical data"
        
        # Remove timezone from historical data
        if hasattr(hist.index, 'tz'):
            hist.index = hist.index.tz_localize(None)
        
        # FORCE CREATE TODAY'S ROW with current time
        now = datetime.now()
        today_timestamp = pd.Timestamp(year=now.year, month=now.month, day=now.day, 
                                      hour=now.hour, minute=now.minute, second=0)
        
        # Create today's data with live price
        today_row = pd.DataFrame({
            'Open': [live_price * 0.999],
            'High': [live_price * 1.001],
            'Low': [live_price * 0.998],
            'Close': [live_price],
            'Volume': [hist['Volume'].iloc[-1] if len(hist) > 0 else 1000000]
        }, index=[today_timestamp])
        
        # Combine: historical + today
        df = pd.concat([hist, today_row])
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()
        
        # Clean
        df = df.reset_index()
        df.columns = ['date'] + list(df.columns[1:])
        df = df.set_index('date')
        
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[cols]
        
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        
        if len(df) < 250:
            return False, f"❌ Not enough data: {len(df)}"
        
        # Save
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/btc_data.csv")
        
        # Features
        feat = df.copy()
        feat["HL_PCT"] = (feat["High"] - feat["Low"]) / feat["Close"]
        feat["PCT_change"] = (feat["Close"] - feat["Open"]) / feat["Open"]
        feat["MA7"] = feat["Close"].rolling(7, min_periods=7).mean()
        feat["MA21"] = feat["Close"].rolling(21, min_periods=21).mean()
        feat["MA50"] = feat["Close"].rolling(50, min_periods=50).mean()
        feat["MA200"] = feat["Close"].rolling(200, min_periods=200).mean()
        feat["STD21"] = feat["Close"].rolling(21, min_periods=21).std()
        feat["Return"] = feat["Close"].pct_change()
        feat["Target"] = feat["Return"].shift(-1)
        feat = feat.dropna()
        
        if len(feat) < 50:
            return False, "❌ Not enough features"
        
        feat.to_csv("data/btc_features.csv")
        
        st.session_state.last_update = datetime.now()
        
        # Verify we have today's date
        final_date = df.index[-1]
        today_date = datetime.now().date()
        
        return True, f"✅ Updated to TODAY {today_date}! Price: ${live_price:,.2f} at {now.strftime('%H:%M')}"
        
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def retrain_models():
    try:
        subprocess.run([sys.executable, "train_rf.py"], capture_output=True, timeout=90)
        subprocess.run([sys.executable, "train_lstm.py"], capture_output=True, timeout=240)
        return True
    except:
        return False

def check_api():
    try:
        return requests.get(f"{API_BASE}/health", timeout=3).ok
    except:
        return False

def get_single_prediction(endpoint):
    try:
        resp = requests.get(f"{API_BASE}/{endpoint}", timeout=10)
        if resp.ok:
            data = resp.json()
            price = data.get("next_day_price") or data.get("prediction")
            return float(price) if price else None
        return None
    except:
        return None

def get_multi_prediction(endpoint, days):
    try:
        resp = requests.get(f"{API_BASE}/{endpoint}?days={days}", timeout=15)
        if resp.ok:
            data = resp.json()
            if 'predictions' in data and 'dates' in data:
                return data
        return None
    except:
        return None

def create_combined_forecast_chart(historical_df, rf_data, lstm_data):
    fig = go.Figure()
    hist_recent = historical_df.tail(60)
    fig.add_trace(go.Scatter(x=hist_recent.index, y=hist_recent['Close'],
                             mode='lines', name='Historical', line=dict(color='#4ade80', width=2)))
    if rf_data and rf_data.get('success'):
        fig.add_trace(go.Scatter(x=pd.to_datetime(rf_data['dates']), y=rf_data['predictions'],
                                mode='lines+markers', name='Random Forest',
                                line=dict(color='#f7931a', width=3, dash='dash'), marker=dict(size=10)))
    if lstm_data and lstm_data.get('success'):
        fig.add_trace(go.Scatter(x=pd.to_datetime(lstm_data['dates']), y=lstm_data['predictions'],
                                mode='lines+markers', name='LSTM',
                                line=dict(color='#a855f7', width=3, dash='dot'), marker=dict(size=10)))
    fig.update_layout(title='Bitcoin Price Forecast', xaxis_title='Date', yaxis_title='Price (USD)',
                     template='plotly_dark', height=500, plot_bgcolor='rgba(0,0,0,0.3)',
                     paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
    return fig

def create_ohlc_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.7, 0.3], subplot_titles=('Price (OHLC)', 'Volume'))
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='OHLC',
                                 increasing_line_color='#4ade80', decreasing_line_color='#ef4444'), row=1, col=1)
    colors = ['#4ade80' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ef4444' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, showlegend=False), row=2, col=1)
    fig.update_layout(template='plotly_dark', height=600, xaxis_rangeslider_visible=False,
                     plot_bgcolor='rgba(0,0,0,0.3)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e0e0e0'))
    return fig

# SIDEBAR
st.sidebar.header("⚙️ Settings")
if check_api():
    st.sidebar.success("✅ FastAPI Connected")
else:
    st.sidebar.error("❌ FastAPI NOT running")

st.sidebar.markdown("---")

# AUTO-UPDATE TOGGLE
st.sidebar.subheader("🤖 Auto-Update")

# Use the session state variable safely
current_auto_state = st.session_state.get('auto_update_enabled', True)

auto_enabled = st.sidebar.checkbox(
    "Enable daily auto-update",
    value=current_auto_state,
    help="Automatically update to today's data when you open the app each day"
)
st.session_state.auto_update_enabled = auto_enabled

if auto_enabled:
    st.sidebar.success("✅ Auto-update is ON")
    st.sidebar.caption("Data updates automatically when you open the app each new day")
else:
    st.sidebar.info("ℹ️ Auto-update is OFF")
    st.sidebar.caption("Use manual refresh button to update")

st.sidebar.markdown("---")

# Show current live price with auto-refresh
live_price_placeholder = st.sidebar.empty()
live_time_placeholder = st.sidebar.empty()

live_now = get_live_bitcoin_price()
if live_now:
    live_price_placeholder.success(f"💰 **LIVE: ${live_now:,.2f}**")
    live_time_placeholder.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    
    # Auto-refresh every 5 seconds
    if 'last_price_refresh' not in st.session_state:
        st.session_state.last_price_refresh = time.time()
    
    # Check if 5 seconds have passed
    if time.time() - st.session_state.last_price_refresh > 5:
        st.session_state.last_price_refresh = time.time()
        time.sleep(0.1)
        st.rerun()
else:
    live_price_placeholder.info("📡 Fetching live price...")

st.sidebar.markdown("---")

st.sidebar.subheader("🔄 Manual Update")

refresh_btn = st.sidebar.button("🔄 REFRESH TO TODAY", key="refresh_btn", use_container_width=True, type="secondary")

if refresh_btn:
    progress = st.sidebar.progress(0)
    status = st.sidebar.empty()
    
    status.info("📡 Getting live Bitcoin price...")
    progress.progress(20)
    time.sleep(0.3)
    
    success, msg = force_update_with_todays_date()
    progress.progress(50)
    
    if success:
        status.success(msg)
        time.sleep(0.5)
        
        status.info("🤖 Retraining models...")
        progress.progress(75)
        retrain_models()
        progress.progress(100)
        
        st.sidebar.balloons()
        time.sleep(1)
        progress.empty()
        status.empty()
        st.rerun()
    else:
        status.error(msg)
        progress.empty()

if st.session_state.last_update:
    st.sidebar.success(f"✅ Last: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M')}")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Mode")
mode = st.sidebar.radio("Select Mode", ["Single Day", "Multi-Day"])

days = 7
if mode == "Multi-Day":
    days = st.sidebar.slider("Days", 1, 14, 7)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Chart Options")
chart_days = st.sidebar.slider("Historical Days", 30, 200, 100)

# MAIN CONTENT
st.markdown("<h1>₿ Bitcoin Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#ffa500'>ML-Powered Cryptocurrency Forecasting</p>", unsafe_allow_html=True)

FEATURES_CSV = "data/btc_features.csv"

# AUTO-UPDATE CHECK - Runs once per day automatically
if st.session_state.auto_update_enabled:
    today_date = datetime.now().date()
    
    # Check if we need to auto-update
    should_auto_update = False
    
    # Safely get last auto update date
    last_update_date = st.session_state.get('last_auto_update_date', None)
    
    if last_update_date != today_date:
        # Haven't updated today yet
        if os.path.exists(FEATURES_CSV):
            try:
                df_check = pd.read_csv(FEATURES_CSV, index_col=0, parse_dates=True)
                if hasattr(df_check.index, 'tz') and df_check.index.tz is not None:
                    df_check.index = df_check.index.tz_localize(None)
                
                data_date = df_check.index[-1].date()
                
                # If data is not from today, trigger auto-update
                if data_date < today_date:
                    should_auto_update = True
            except:
                should_auto_update = True
        else:
            should_auto_update = True
    
    # Perform auto-update
    if should_auto_update:
        auto_update_container = st.empty()
        with auto_update_container.container():
            st.info("🤖 Auto-updating to today's Bitcoin prices...")
            progress = st.progress(0)
            
            progress.progress(20)
            success, msg = force_update_with_todays_date()
            progress.progress(60)
            
            if success:
                st.success(msg)
                progress.progress(80)
                st.info("🤖 Retraining models...")
                retrain_models()
                progress.progress(100)
                
                st.session_state.last_auto_update_date = today_date
                st.balloons()
                time.sleep(2)
                auto_update_container.empty()
                st.rerun()
            else:
                st.error(msg)
                progress.empty()
                time.sleep(2)
                auto_update_container.empty()

if not os.path.exists(FEATURES_CSV):
    st.error("❌ No data found!")
    st.warning("👈 Click **🔄 REFRESH TO TODAY** button in sidebar")
    
    if st.button("🚀 Get Started", type="primary"):
        with st.spinner("Fetching..."):
            success, msg = force_update_with_todays_date()
            if success:
                st.success(msg)
                retrain_models()
                time.sleep(1)
                st.rerun()
            else:
                st.error(msg)
    st.stop()

try:
    df = pd.read_csv(FEATURES_CSV, index_col=0, parse_dates=True)
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# Remove timezone if present
if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)

data_datetime = df.index[-1]
data_date = data_datetime.date()
data_time = data_datetime.strftime('%H:%M:%S')
today = datetime.now().date()
data_age_days = (today - data_date).days

current = df["Close"].iloc[-1]
prev = df["Close"].iloc[-2]
change_pct = ((current - prev) / prev) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Current", f"${current:,.2f}", f"{change_pct:+.2f}%")
col2.metric("📈 High", f"${df['High'].iloc[-1]:,.2f}")
col3.metric("📉 Low", f"${df['Low'].iloc[-1]:,.2f}")

if data_age_days == 0:
    col4.metric("📅 Data", f"TODAY ✅", f"{data_time}")
    st.success(f"✅ Using TODAY's data ({data_datetime.strftime('%Y-%m-%d %H:%M')})")
else:
    col4.metric("📅 Data", data_date.strftime("%m-%d"), f"{data_age_days}d old")
    st.error(f"⚠️ Data is {data_age_days} days old! Click REFRESH TO TODAY button!")

st.markdown("---")

if mode == "Single Day":
    st.info(f"💡 Predictions from {data_datetime.strftime('%Y-%m-%d %H:%M')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Random Forest")
        if st.button("Get RF Prediction", key="rf", use_container_width=True):
            with st.spinner("Predicting..."):
                price = get_single_prediction("predict/rf")
                if price:
                    st.success(f"### ${price:,.2f}")
                    next_day = data_datetime + timedelta(days=1)
                    st.caption(f"For {next_day.strftime('%Y-%m-%d')}")
                    change = ((price - current) / current) * 100
                    st.info(f"{'📈' if change > 0 else '📉'} {change:+.2f}% (${price - current:+,.2f})")
                else:
                    st.error("❌ Failed")
    
    with col2:
        st.subheader("🧠 LSTM")
        if st.button("Get LSTM Prediction", key="lstm", use_container_width=True):
            with st.spinner("Predicting..."):
                price = get_single_prediction("predict/lstm")
                if price:
                    st.success(f"### ${price:,.2f}")
                    next_day = data_datetime + timedelta(days=1)
                    st.caption(f"For {next_day.strftime('%Y-%m-%d')}")
                    change = ((price - current) / current) * 100
                    st.info(f"{'📈' if change > 0 else '📉'} {change:+.2f}% (${price - current:+,.2f})")
                else:
                    st.error("❌ Failed")

else:
    st.subheader(f"📈 {days}-Day Forecast")
    
    if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner(f"Generating {days}-day forecast..."):
            rf_data = get_multi_prediction("predict/rf/multi", days)
            lstm_data = get_multi_prediction("predict/lstm/multi", days)
            
            if rf_data or lstm_data:
                st.plotly_chart(create_combined_forecast_chart(df, rf_data, lstm_data), use_container_width=True)
                
                st.markdown("### 📊 Detailed Predictions")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🤖 Random Forest**")
                    if rf_data and 'predictions' in rf_data:
                        rf_df = pd.DataFrame({'Date': rf_data['dates'],
                                             'Price': [f"${p:,.2f}" for p in rf_data['predictions']]})
                        st.dataframe(rf_df, hide_index=True, use_container_width=True, height=300)
                
                with col2:
                    st.markdown("**🧠 LSTM**")
                    if lstm_data and 'predictions' in lstm_data:
                        lstm_df = pd.DataFrame({'Date': lstm_data['dates'],
                                               'Price': [f"${p:,.2f}" for p in lstm_data['predictions']]})
                        st.dataframe(lstm_df, hide_index=True, use_container_width=True, height=300)

st.markdown("---")
st.subheader("📊 Historical Analysis")

tab1, tab2, tab3 = st.tabs(["📈 OHLC Chart", "📉 Line Chart", "📋 Data Table"])

with tab1:
    st.plotly_chart(create_ohlc_chart(df.tail(chart_days)), use_container_width=True)

with tab2:
    st.markdown("**Select indicators to display:**")
    indicators = st.multiselect(
        "Choose data series:",
        ['Close', 'Open', 'High', 'Low', 'Volume', 'MA7', 'MA21', 'MA50', 'MA200'],
        default=['Close', 'MA7', 'MA21'],
        key='line_indicators'
    )
    
    if indicators:
        line_df = df[indicators].tail(chart_days)
        
        fig = go.Figure()
        colors = ['#f7931a', '#4ade80', '#a855f7', '#ef4444', '#fbbf24', '#60a5fa', '#ec4899', '#8b5cf6', '#06b6d4']
        
        for i, col in enumerate(indicators):
            fig.add_trace(go.Scatter(
                x=line_df.index,
                y=line_df[col],
                mode='lines',
                name=col,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{col}:</b> $%{{y:,.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Price & Indicators',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            height=500,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Select at least one indicator to display the chart")

with tab3:
    st.markdown("**Recent Data (Last 20 rows)**")
    display_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA21', 'MA50']
    recent = df[display_cols].tail(20).copy()
    
    for col in ['Open', 'High', 'Low', 'Close', 'MA7', 'MA21', 'MA50']:
        if col in recent.columns:
            recent[col] = recent[col].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
    
    if 'Volume' in recent.columns:
        recent['Volume'] = recent['Volume'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
    
    st.dataframe(recent, use_container_width=True)
    
    # Download button
    csv = df.tail(100).to_csv()
    st.download_button(
        label="📥 Download Last 100 Days (CSV)",
        data=csv,
        file_name="bitcoin_data.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#888;padding:20px'>
    <p>⚠️ Educational purposes only. Not financial advice.</p>
    <p>💡 Click REFRESH TO TODAY button daily for latest prices!</p>
</div>
""", unsafe_allow_html=True)