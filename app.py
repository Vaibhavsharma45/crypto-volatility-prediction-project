import gradio as gr
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Load model
def load_model():
    with open('model/volatility_model.pkl', 'rb') as f:
        return pickle.load(f)

model_package = load_model()
model = model_package['model']
scaler = model_package['scaler']
feature_columns = model_package['feature_columns']

# Feature engineering function (same as before)
def create_features(df):
    df = df.sort_values('date').reset_index(drop=True)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_14d'] = df['log_return'].rolling(window=14).std()
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_14'] = df['close'].rolling(window=14).mean()
    df['ma_30'] = df['close'].rolling(window=30).mean()
    
    bb_middle = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    df['liquidity_ratio'] = df['volume'] / (df['marketCap'] + 1)
    df['momentum_7'] = df['close'] / df['close'].shift(7)
    
    return df.dropna()

# Main prediction function
def predict_volatility(file, crypto_name):
    try:
        # Load CSV
        df = pd.read_csv(file.name)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter by crypto
        crypto_df = df[df['crypto_name'] == crypto_name].copy()
        
        if len(crypto_df) == 0:
            return "Error: Cryptocurrency not found!", None, None
        
        # Create features
        crypto_df = create_features(crypto_df)
        
        # Prepare features
        X = crypto_df[feature_columns]
        X_scaled = scaler.transform(X)
        
        # Predict
        predictions = model.predict(X_scaled)
        crypto_df['predicted_volatility'] = predictions
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=crypto_df['date'],
            y=crypto_df['predicted_volatility'],
            mode='lines',
            name='Predicted Volatility',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title=f'{crypto_name} - 7-Day Rolling Volatility Prediction',
            xaxis_title='Date',
            yaxis_title='Volatility',
            height=500
        )
        
        # Statistics
        stats = f"""
        📊 Volatility Statistics for {crypto_name}:
        
        Mean Volatility: {predictions.mean():.6f}
        Max Volatility: {predictions.max():.6f}
        Min Volatility: {predictions.min():.6f}
        Std Volatility: {predictions.std():.6f}
        
        Total Predictions: {len(predictions):,}
        """
        
        # Prepare download data
        download_df = crypto_df[['date', 'close', 'predicted_volatility']]
        
        return stats, fig, download_df
        
    except Exception as e:
        return f"Error: {str(e)}", None, None

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Crypto Volatility Predictor") as demo:
    gr.Markdown(
        """
        # 📈 Cryptocurrency Volatility Prediction
        ### Machine Learning Model for 7-Day Rolling Volatility Prediction
        
        Upload your cryptocurrency market data and get volatility predictions!
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 Upload Data")
            file_input = gr.File(
                label="Upload CSV File",
                file_types=[".csv"]
            )
            
            crypto_input = gr.Textbox(
                label="Cryptocurrency Name",
                placeholder="e.g., Bitcoin, Ethereum",
                value="Bitcoin"
            )
            
            predict_btn = gr.Button("🔮 Predict Volatility", variant="primary")
            
            gr.Markdown(
                """
                ### 📋 Required CSV Format:
                - date
                - crypto_name
                - open, high, low, close
                - volume, marketCap
                """
            )
    
    with gr.Row():
        with gr.Column():
            stats_output = gr.Textbox(
                label="📊 Statistics",
                lines=10
            )
    
    with gr.Row():
        plot_output = gr.Plot(label="📈 Volatility Prediction Chart")
    
    with gr.Row():
        download_output = gr.File(label="💾 Download Predictions")
    
    # Button click handler
    predict_btn.click(
        fn=predict_volatility,
        inputs=[file_input, crypto_input],
        outputs=[stats_output, plot_output, download_output]
    )
    
    gr.Markdown(
        """
        ---
        ### 📖 About This Model
        - **Algorithm:** Random Forest Regressor
        - **Features:** 14 technical indicators
        - **Performance:** R² Score ~0.85
        - **Training Data:** Historical cryptocurrency market data
        
        **Developed for PW Skills | Machine Learning Project**
        """
    )

# Launch
if __name__ == "__main__":
    demo.launch()