"""
Streamlit Deployment App with Auto Model Training
Interactive cryptocurrency volatility prediction web application
Author: PW Skills Student
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import os
from pathlib import Path
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
    </style>
""", unsafe_allow_html=True)

# Function to train model if not exists
def train_model_if_needed():
    """Train model if it doesn't exist"""
    model_path = 'model/volatility_model.pkl'
    
    if not os.path.exists(model_path):
        st.warning("⚠️ Model not found. Training model now... This will take a few minutes.")
        
        try:
            # Import required libraries
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Create sample model for deployment
            # In production, you would load actual training data here
            st.info("📊 Creating model with default parameters...")
            
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            
            scaler = StandardScaler()
            
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume', 'marketCap',
                'log_return', 'volatility_14d', 
                'ma_7', 'ma_14', 'ma_30',
                'bb_width', 'atr', 
                'liquidity_ratio', 'momentum_7'
            ]
            
            # Create model directory if it doesn't exist
            os.makedirs('model', exist_ok=True)
            
            # Package model
            model_package = {
                'model': model,
                'scaler': scaler,
                'feature_columns': feature_columns,
                'best_params': None
            }
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model_package, f)
            
            st.success("✅ Model created successfully!")
            st.info("ℹ️ Note: This is a placeholder model. Upload data to train with actual data.")
            
        except Exception as e:
            st.error(f"❌ Error creating model: {str(e)}")
            return None
    
    return model_path

# Load model function with caching
@st.cache_resource
def load_model():
    """Load trained model and components"""
    try:
        # Check if model exists, if not create it
        model_path = train_model_if_needed()
        
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            return model_package
        else:
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
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

# Train model with uploaded data
def train_model_with_data(df, feature_columns):
    """Train model with uploaded data"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        
        st.info("🔧 Training model with uploaded data...")
        
        # Create features
        df_features = create_features(df)
        
        if len(df_features) < 100:
            st.error("❌ Not enough data after feature engineering. Need at least 100 rows.")
            return None
        
        # Calculate target
        df_features['volatility_7d'] = df_features['log_return'].rolling(window=7).std()
        df_features = df_features.dropna()
        
        # Prepare features and target
        X = df_features[feature_columns]
        y = df_features['volatility_7d']
        
        # Split data (80/20)
        split_idx = int(len(df_features) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Save model
        model_package = {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'best_params': None
        }
        
        os.makedirs('model', exist_ok=True)
        with open('model/volatility_model.pkl', 'wb') as f:
            pickle.dump(model_package, f)
        
        st.success("✅ Model trained successfully with your data!")
        
        # Clear cache to reload new model
        st.cache_resource.clear()
        
        return model_package
        
    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        return None

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
        st.error("❌ Failed to load or create model. Please check the error messages above.")
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
    
    # Option to train with uploaded data
    train_with_data = st.sidebar.checkbox(
        "🔧 Train model with uploaded data",
        help="Enable this to train the model with your uploaded data"
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
            
            # Train model if requested
            if train_with_data:
                if st.sidebar.button("🚀 Train Model Now"):
                    with st.spinner("Training model... This may take a few minutes..."):
                        new_model_package = train_model_with_data(df, feature_columns)
                        if new_model_package:
                            model = new_model_package['model']
                            scaler = new_model_package['scaler']
            
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
                st.metric("📅 Total Records", f"{len(crypto_df):,}")
            
            with col2:
                latest_price = crypto_df['close'].iloc[-1]
                st.metric("💵 Latest Price", f"${latest_price:,.2f}")
            
            with col3:
                latest_volume = crypto_df['volume'].iloc[-1]
                st.metric("📊 24h Volume", f"${latest_volume:,.0f}")
            
            with col4:
                latest_mcap = crypto_df['marketCap'].iloc[-1]
                st.metric("🏦 Market Cap", f"${latest_mcap:,.0f}")
            
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
                        
                        if len(crypto_df) == 0:
                            st.error("❌ Not enough data after feature engineering. Need more historical data.")
                            st.stop()
                        
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
                            st.metric("Mean Volatility", f"{predictions.mean():.6f}")
                        
                        with col2:
                            st.metric("Max Volatility", f"{predictions.max():.6f}")
                        
                        with col3:
                            st.metric("Min Volatility", f"{predictions.min():.6f}")
                        
                        with col4:
                            st.metric("Std Volatility", f"{predictions.std():.6f}")
                        
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
        This application uses a **Random Forest Regressor** to predict 7-day rolling volatility.
        
        **Features Used:**
        - Price data (OHLC)
        - Volume and Market Cap
        - Technical indicators (MA, BB, ATR)
        - Log returns and momentum
        
        **Deployment Note:**
        - If model file is not found, a default model is created
        - You can train the model with your own data using the checkbox in sidebar
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