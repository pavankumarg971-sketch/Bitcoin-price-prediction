# app_streamlit.py - Enhanced with multi-day predictions and advanced charts
import streamlit as st
import requests
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_BASE = "http://127.0.0.1:8000"

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    h1 {
        color: #f7931a !important;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    h2, h3 { color: #ffa500 !important; }
    .stButton > button {
        background: linear-gradient(90deg, #f7931a 0%, #ffa500 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(247, 147, 26, 0.4);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 165, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

def check_api():
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        return resp.ok
    except:
        try:
            resp = requests.get(f"{API_BASE}/", timeout=3)
            return resp.ok
        except:
            return False

def get_single_prediction(endpoint):
    """Get single prediction"""
    try:
        resp = requests.get(f"{API_BASE}/{endpoint}", timeout=10)
        if resp.ok:
            data = resp.json()
            price = data.get("next_day_price") or data.get("prediction")
            if price is not None:
                return float(price)
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def get_multi_prediction(endpoint, days):
    """Get multi-day prediction"""
    try:
        url = f"{API_BASE}/{endpoint}?days={days}"
        resp = requests.get(url, timeout=15)
        
        if resp.ok:
            data = resp.json()
            # Check if we have predictions and dates
            if 'predictions' in data and 'dates' in data:
                return data
            else:
                st.error(f"API returned incomplete data: {list(data.keys())}")
                return None
        else:
            st.error(f"HTTP Error {resp.status_code}")
            return None
    except Exception as e:
        st.error(f"Request Error: {str(e)}")
        return None

def create_combined_forecast_chart(historical_df, rf_data, lstm_data):
    """Create an interactive Plotly chart with historical + predictions"""
    fig = go.Figure()
    
    # Historical data (last 60 days)
    hist_recent = historical_df.tail(60)
    fig.add_trace(go.Scatter(
        x=hist_recent.index,
        y=hist_recent['Close'],
        mode='lines',
        name='Historical',
        line=dict(color='#4ade80', width=2),
        hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # RF Predictions
    if rf_data and rf_data.get('success'):
        rf_dates = pd.to_datetime(rf_data['dates'])
        fig.add_trace(go.Scatter(
            x=rf_dates,
            y=rf_data['predictions'],
            mode='lines+markers',
            name='Random Forest',
            line=dict(color='#f7931a', width=3, dash='dash'),
            marker=dict(size=10, symbol='circle'),
            hovertemplate='<b>Date:</b> %{x}<br><b>RF Prediction:</b> $%{y:,.2f}<extra></extra>'
        ))
    
    # LSTM Predictions
    if lstm_data and lstm_data.get('success'):
        lstm_dates = pd.to_datetime(lstm_data['dates'])
        fig.add_trace(go.Scatter(
            x=lstm_dates,
            y=lstm_data['predictions'],
            mode='lines+markers',
            name='LSTM',
            line=dict(color='#a855f7', width=3, dash='dot'),
            marker=dict(size=10, symbol='diamond'),
            hovertemplate='<b>Date:</b> %{x}<br><b>LSTM Prediction:</b> $%{y:,.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Bitcoin Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0,0,0,0.3)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0', size=12)
    )
    
    return fig

def create_ohlc_chart(df):
    """Create OHLC candlestick chart with volume"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price (OHLC)', 'Volume')
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        increasing_line_color='#4ade80',
        decreasing_line_color='#ef4444'
    ), row=1, col=1)
    
    # Volume bars
    colors = ['#4ade80' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ef4444' 
              for i in range(len(df))]
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker_color=colors,
        showlegend=False
    ), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        height=600,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        plot_bgcolor='rgba(0,0,0,0.3)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0')
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    
    return fig

# Page config
st.set_page_config(page_title="Bitcoin Prediction", page_icon="₿", layout="wide")

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    if check_api():
        st.success("✅ FastAPI Connected")
    else:
        st.error("❌ FastAPI NOT running")
    
    st.markdown("---")
    mode = st.radio("📊 Mode", ["Single Day", "Multi-Day"])
    
    days = 7
    if mode == "Multi-Day":
        days = st.slider("Days to Predict", 1, 14, 7)
    
    st.markdown("---")
    st.markdown("### 📈 Chart Options")
    show_volume = st.checkbox("Show Volume", value=True)
    chart_days = st.slider("Historical Days", 30, 200, 100)
    
    st.markdown("---")
    st.info("💡 **Tip:** Use Multi-Day mode for weekly forecasts")

# Header
st.markdown("<h1>₿ Bitcoin Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#ffa500'>ML-Powered Cryptocurrency Forecasting</p>", unsafe_allow_html=True)

# Load data
FEATURES_CSV = "data/btc_features.csv"
df = None

if os.path.exists(FEATURES_CSV):
    df = pd.read_csv(FEATURES_CSV, index_col=0, parse_dates=True)
    
    # Metrics
    current = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2]
    change_pct = ((current - prev) / prev) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Current", f"${current:,.2f}", f"{change_pct:+.2f}%")
    col2.metric("📈 High", f"${df['High'].iloc[-1]:,.2f}")
    col3.metric("📉 Low", f"${df['Low'].iloc[-1]:,.2f}")

st.markdown("---")

# Predictions
if mode == "Single Day":
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Random Forest")
        if st.button("Get RF Prediction", key="rf"):
            with st.spinner("Predicting..."):
                price = get_single_prediction("predict/rf")
                if price is not None:
                    st.success(f"### ${price:,.2f}")
                    st.caption("Next-day closing price prediction")
                else:
                    st.error("Prediction failed")
    
    with col2:
        st.subheader("🧠 LSTM")
        if st.button("Get LSTM Prediction", key="lstm"):
            with st.spinner("Predicting..."):
                price = get_single_prediction("predict/lstm")
                if price is not None:
                    st.success(f"### ${price:,.2f}")
                    st.caption("Next-day closing price prediction")
                else:
                    st.error("Prediction failed")

else:  # Multi-Day
    st.subheader(f"📈 {days}-Day Forecast")
    
    if st.button("🔮 Generate Forecast", type="primary", key="multi"):
        with st.spinner(f"Generating {days}-day forecast..."):
            rf_data = get_multi_prediction("predict/rf/multi", days)
            lstm_data = get_multi_prediction("predict/lstm/multi", days)
            
            if (rf_data and 'predictions' in rf_data) or (lstm_data and 'predictions' in lstm_data):
                # Combined forecast chart
                if df is not None:
                    st.plotly_chart(
                        create_combined_forecast_chart(df, rf_data, lstm_data),
                        use_container_width=True
                    )
                
                # Prediction tables side by side
                st.markdown("### 📊 Detailed Predictions")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🤖 Random Forest Forecast**")
                    if rf_data:
                        # Create simple table with Date | Price
                        rf_table = pd.DataFrame({
                            'Date': rf_data['dates'],
                            'Predicted Price': [f"${p:,.2f}" for p in rf_data['predictions']]
                        })
                        st.dataframe(rf_table, hide_index=True, use_container_width=True, height=300)
                        
                        # Summary stats
                        avg_price = sum(rf_data['predictions']) / len(rf_data['predictions'])
                        min_price = min(rf_data['predictions'])
                        max_price = max(rf_data['predictions'])
                        
                        st.markdown(f"""
                        📊 **Summary:**
                        - Average: ${avg_price:,.2f}
                        - Minimum: ${min_price:,.2f}
                        - Maximum: ${max_price:,.2f}
                        """)
                    else:
                        st.warning("⚠️ RF predictions unavailable")
                
                with col2:
                    st.markdown("**🧠 LSTM Forecast**")
                    if lstm_data:
                        # Create simple table with Date | Price
                        lstm_table = pd.DataFrame({
                            'Date': lstm_data['dates'],
                            'Predicted Price': [f"${p:,.2f}" for p in lstm_data['predictions']]
                        })
                        st.dataframe(lstm_table, hide_index=True, use_container_width=True, height=300)
                        
                        # Summary stats
                        avg_price = sum(lstm_data['predictions']) / len(lstm_data['predictions'])
                        min_price = min(lstm_data['predictions'])
                        max_price = max(lstm_data['predictions'])
                        
                        st.markdown(f"""
                        📊 **Summary:**
                        - Average: ${avg_price:,.2f}
                        - Minimum: ${min_price:,.2f}
                        - Maximum: ${max_price:,.2f}
                        """)
                    else:
                        st.warning("⚠️ LSTM predictions unavailable")
            else:
                st.error("Both models failed to generate predictions. Check API logs.")

# Historical Analysis
if df is not None:
    st.markdown("---")
    st.subheader("📊 Historical Price Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 OHLC Chart", "📉 Line Chart", "📋 Data Table"])
    
    with tab1:
        # Candlestick chart
        chart_df = df.tail(chart_days)
        st.plotly_chart(create_ohlc_chart(chart_df), use_container_width=True)
    
    with tab2:
        # Multi-line chart
        st.markdown("**Select indicators to display:**")
        indicators = st.multiselect(
            "Choose data series:",
            ['Close', 'Open', 'High', 'Low', 'MA7', 'MA21', 'MA50'],
            default=['Close', 'MA21'],
            key='line_chart_select'
        )
        
        if indicators:
            line_df = df[indicators].tail(chart_days)
            
            fig = go.Figure()
            colors = ['#f7931a', '#4ade80', '#a855f7', '#ef4444', '#fbbf24', '#60a5fa', '#ec4899']
            
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
                template='plotly_dark',
                height=400,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0.3)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e0e0e0')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Data table
        st.markdown("**Recent Data (Last 20 rows)**")
        display_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        recent_data = df[display_cols].tail(20)
        
        # Format the dataframe
        formatted_data = recent_data.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            formatted_data[col] = formatted_data[col].apply(lambda x: f"${x:,.2f}")
        formatted_data['Volume'] = formatted_data['Volume'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(formatted_data, use_container_width=True)
        
        # Download button
        csv = df.tail(100).to_csv()
        st.download_button(
            label="📥 Download Last 100 Days (CSV)",
            data=csv,
            file_name="bitcoin_data.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#888;padding:20px'>
    <p>⚠️ <strong>Disclaimer:</strong> Educational purposes only. Not financial advice.</p>
    <p>Built with ❤️ using FastAPI + Streamlit + TensorFlow + Scikit-learn</p>
    <p style='font-size:12px;color:#666'>Last Updated: November 2025</p>
</div>
""", unsafe_allow_html=True)