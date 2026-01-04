"""
Streamlit Deployment App
Interactive cryptocurrency volatility prediction web application
Author: PW Skills Student
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crypto Volatility Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Load model function with caching
@st.cache_resource
def load_model():
    """Load trained model and components"""
    try:
        with open('model/volatility_model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error("❌ Model file not found! Please train the model first.")
        return None

# Feature engineering functions
def create_features(df):
    """
    Create all required features for prediction
    Must match the features used during training
    """
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. Log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Rolling volatility (14-day)
    df['volatility_14d'] = df['log_return'].rolling(window=14).std()
    
    # 3. Moving averages
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_14'] = df['close'].rolling(window=14).mean()
    df['ma_30'] = df['close'].rolling(window=30).mean()
    
    # 4. Bollinger Bands
    bb_middle = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    
    # 5. Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # 6. Liquidity ratio
    df['liquidity_ratio'] = df['volume'] / (df['marketCap'] + 1)
    
    # 7. Momentum
    df['momentum_7'] = df['close'] / df['close'].shift(7)
    
    # Remove NaN values
    return df.dropna()

# Main application
def main():
    # Header
    st.markdown('<p class="main-header">📈 Cryptocurrency Volatility Predictor</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning Model for 7-Day Rolling Volatility Prediction</p>', 
                unsafe_allow_html=True)
    
    # Load model
    model_package = load_model()
    
    if model_package is None:
        st.stop()
    
    model = model_package['model']
    scaler = model_package['scaler']
    feature_columns = model_package['feature_columns']
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar
    st.sidebar.header("📁 Data Upload")
    st.sidebar.markdown("Upload your cryptocurrency market data in CSV format")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload CSV with columns: date, crypto_name, open, high, low, close, volume, marketCap"
    )
    
    # Sample data format
    with st.sidebar.expander("📋 Required CSV Format"):
        sample_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'crypto_name': ['Bitcoin', 'Bitcoin'],
            'open': [42000, 42500],
            'high': [43000, 43500],
            'low': [41500, 42000],
            'close': [42500, 43000],
            'volume': [1000000, 1100000],
            'marketCap': [800000000, 810000000]
        })
        st.dataframe(sample_df, use_container_width=True)
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            df['date'] = pd.to_datetime(df['date'])
            
            st.sidebar.success(f"✅ Data loaded: {len(df):,} rows")
            
            # Data validation
            required_cols = ['date', 'crypto_name', 'open', 'high', 'low', 'close', 'volume', 'marketCap']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.stop()
            
            # Cryptocurrency selection
            cryptos = sorted(df['crypto_name'].unique())
            selected_crypto = st.sidebar.selectbox(
                "🪙 Select Cryptocurrency", 
                cryptos,
                help="Choose which cryptocurrency to analyze"
            )
            
            # Filter data for selected crypto
            crypto_df = df[df['crypto_name'] == selected_crypto].copy()
            
            # Data overview
            st.header(f"📊 {selected_crypto} - Market Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "📅 Total Records", 
                    f"{len(crypto_df):,}",
                    help="Number of data points available"
                )
            
            with col2:
                latest_price = crypto_df['close'].iloc[-1]
                st.metric(
                    "💵 Latest Price", 
                    f"${latest_price:,.2f}",
                    help="Most recent closing price"
                )
            
            with col3:
                latest_volume = crypto_df['volume'].iloc[-1]
                st.metric(
                    "📊 24h Volume", 
                    f"${latest_volume:,.0f}",
                    help="Latest trading volume"
                )
            
            with col4:
                latest_mcap = crypto_df['marketCap'].iloc[-1]
                st.metric(
                    "🏦 Market Cap", 
                    f"${latest_mcap:,.0f}",
                    help="Latest market capitalization"
                )
            
            # Price chart
            st.subheader("📈 Price History")
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=crypto_df['date'],
                y=crypto_df['close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            fig_price.update_layout(
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Prediction section
            st.header("🔮 Volatility Prediction")
            
            if st.button("🚀 Predict Volatility", type="primary", use_container_width=True):
                with st.spinner("Creating features and making predictions..."):
                    try:
                        # Create features
                        crypto_df = create_features(crypto_df)
                        
                        # Prepare features
                        X = crypto_df[feature_columns]
                        X_scaled = scaler.transform(X)
                        
                        # Make predictions
                        predictions = model.predict(X_scaled)
                        crypto_df['predicted_volatility'] = predictions
                        
                        st.success("✅ Predictions completed successfully!")
                        
                        # Volatility statistics
                        st.subheader("📊 Volatility Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Mean Volatility",
                                f"{predictions.mean():.6f}",
                                help="Average predicted volatility"
                            )
                        
                        with col2:
                            st.metric(
                                "Max Volatility",
                                f"{predictions.max():.6f}",
                                help="Maximum predicted volatility"
                            )
                        
                        with col3:
                            st.metric(
                                "Min Volatility",
                                f"{predictions.min():.6f}",
                                help="Minimum predicted volatility"
                            )
                        
                        with col4:
                            st.metric(
                                "Std Volatility",
                                f"{predictions.std():.6f}",
                                help="Standard deviation of volatility"
                            )
                        
                        # Volatility chart
                        st.subheader("📈 Predicted Volatility Over Time")
                        
                        fig_vol = go.Figure()
                        fig_vol.add_trace(go.Scatter(
                            x=crypto_df['date'],
                            y=crypto_df['predicted_volatility'],
                            mode='lines',
                            name='Predicted Volatility',
                            line=dict(color='#ff7f0e', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(255, 127, 14, 0.2)'
                        ))
                        
                        fig_vol.update_layout(
                            xaxis_title="Date",
                            yaxis_title="7-Day Rolling Volatility",
                            hovermode='x unified',
                            height=450
                        )
                        
                        st.plotly_chart(fig_vol, use_container_width=True)
                        
                        # Distribution
                        st.subheader("📊 Volatility Distribution")
                        
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Histogram(
                            x=predictions,
                            nbinsx=50,
                            name='Volatility',
                            marker_color='#2ca02c'
                        ))
                        
                        fig_dist.update_layout(
                            xaxis_title="Volatility",
                            yaxis_title="Frequency",
                            height=400
                        )
                        
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Data table
                        st.subheader("📋 Prediction Results")
                        
                        display_df = crypto_df[['date', 'close', 'predicted_volatility']].tail(20)
                        display_df.columns = ['Date', 'Close Price', 'Predicted Volatility']
                        
                        st.dataframe(
                            display_df.style.format({
                                'Close Price': '${:,.2f}',
                                'Predicted Volatility': '{:.8f}'
                            }),
                            use_container_width=True
                        )
                        
                        # Download section
                        st.subheader("💾 Download Predictions")
                        
                        csv = crypto_df[['date', 'close', 'predicted_volatility']].to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"{selected_crypto}_volatility_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.exception(e)
    
    else:
        # Instructions when no file is uploaded
        st.info("👈 Please upload a CSV file to get started")
        
        st.markdown("---")
        
        st.header("ℹ️ How to Use")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📤 Step 1: Upload Data
            - Upload your CSV file containing cryptocurrency market data
            - Ensure it has all required columns
            - Multiple cryptocurrencies can be in one file
            """)
            
            st.markdown("""
            ### 🪙 Step 2: Select Cryptocurrency
            - Choose which crypto to analyze from the dropdown
            - View market overview and statistics
            """)
        
        with col2:
            st.markdown("""
            ### 🔮 Step 3: Predict
            - Click the "Predict Volatility" button
            - View predicted volatility trends
            - Analyze statistics and distributions
            """)
            
            st.markdown("""
            ### 💾 Step 4: Download
            - Download predictions as CSV
            - Use for further analysis or reporting
            """)
        
        st.markdown("---")
        
        st.header("📖 About This Model")
        
        st.markdown("""
        This application uses a **Random Forest Regressor** trained on historical cryptocurrency data to predict 7-day rolling volatility.
        
        **Features Used:**
        - Price data (OHLC)
        - Volume and Market Cap
        - Technical indicators (MA, BB, ATR)
        - Log returns and momentum
        
        **Model Performance:**
        - Trained on 80% of historical data
        - Evaluated using RMSE, MAE, and R² metrics
        - Hyperparameter tuned for optimal performance
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Developed for PW Skills | Cryptocurrency Volatility Prediction Project</p>
            <p>© 2025 | Machine Learning & Data Science</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()