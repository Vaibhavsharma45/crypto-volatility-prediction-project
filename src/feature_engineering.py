"""
Feature Engineering Module
Creates technical indicators and target variable for volatility prediction
Author: PW Skills Student
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Creates advanced features for cryptocurrency volatility prediction
    Implements technical indicators used in financial analysis
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.feature_names = []
    
    def load_data(self, filepath):
        """
        Load cleaned data from preprocessing step
        
        Args:
            filepath (str): Path to cleaned CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        print("=" * 60)
        print("LOADING CLEANED DATA")
        print("=" * 60)
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Data loaded: {df.shape[0]:,} rows")
        print(f"✓ Cryptocurrencies: {df['crypto_name'].nunique()}")
        return df
    
    def create_log_returns(self, df):
        """
        Calculate logarithmic returns
        Log returns are preferred in financial analysis as they are:
        - Time-additive
        - Symmetric
        - More suitable for statistical modeling
        
        Formula: log(price_t / price_t-1)
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with log_return column
        """
        print("\n📊 Creating log returns...")
        df['log_return'] = df.groupby('crypto_name')['close'].transform(
            lambda x: np.log(x / x.shift(1))
        )
        print("✓ Log returns created")
        return df
    
    def create_rolling_volatility(self, df):
        """
        Create rolling volatility features
        Volatility = Standard deviation of log returns
        
        Features created:
        - volatility_7d: 7-day rolling volatility (TARGET VARIABLE)
        - volatility_14d: 14-day rolling volatility (feature)
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with volatility columns
        """
        print("\n📈 Creating rolling volatility...")
        
        # 7-day volatility (TARGET VARIABLE)
        df['volatility_7d'] = df.groupby('crypto_name')['log_return'].transform(
            lambda x: x.rolling(window=7).std()
        )
        print("✓ 7-day volatility (TARGET) created")
        
        # 14-day volatility (FEATURE)
        df['volatility_14d'] = df.groupby('crypto_name')['log_return'].transform(
            lambda x: x.rolling(window=14).std()
        )
        print("✓ 14-day volatility created")
        
        return df
    
    def create_moving_averages(self, df):
        """
        Create Simple Moving Averages (SMA)
        MA helps identify trends and support/resistance levels
        
        Features created:
        - ma_7: 7-day moving average
        - ma_14: 14-day moving average
        - ma_30: 30-day moving average
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with MA columns
        """
        print("\n📉 Creating moving averages...")
        
        for window in [7, 14, 30]:
            df[f'ma_{window}'] = df.groupby('crypto_name')['close'].transform(
                lambda x: x.rolling(window=window).mean()
            )
            print(f"✓ MA-{window} created")
        
        return df
    
    def create_bollinger_bands(self, df):
        """
        Create Bollinger Bands indicator
        Bollinger Bands measure market volatility
        
        Formula:
        - Middle Band = 20-day SMA
        - Upper Band = Middle Band + (2 × standard deviation)
        - Lower Band = Middle Band - (2 × standard deviation)
        - BB Width = (Upper Band - Lower Band) / Middle Band
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with Bollinger Band features
        """
        print("\n🎯 Creating Bollinger Bands...")
        
        # Calculate components by group
        df['bb_middle'] = df.groupby('crypto_name')['close'].transform(
            lambda x: x.rolling(window=20).mean()
        )
        
        df['bb_std'] = df.groupby('crypto_name')['close'].transform(
            lambda x: x.rolling(window=20).std()
        )
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (2 * df['bb_std'])
        
        # Calculate Bollinger Band width (normalized)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Drop intermediate columns to keep dataset clean
        df.drop(['bb_middle', 'bb_std', 'bb_upper', 'bb_lower'], axis=1, inplace=True)
        
        print("✓ Bollinger Band width created")
        
        return df
    
    def create_atr(self, df):
        """
        Create Average True Range (ATR)
        ATR measures market volatility by calculating the average of true ranges
        
        True Range = max of:
        1. High - Low
        2. |High - Previous Close|
        3. |Low - Previous Close|
        
        ATR = 14-day average of True Range
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with ATR column
        """
        print("\n💹 Creating Average True Range (ATR)...")
        
        # Calculate three components of True Range
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df.groupby('crypto_name')['close'].shift(1))
        df['low_close'] = np.abs(df['low'] - df.groupby('crypto_name')['close'].shift(1))
        
        # True Range = maximum of the three components
        df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # ATR = 14-day average of True Range
        df['atr'] = df.groupby('crypto_name')['true_range'].transform(
            lambda x: x.rolling(window=14).mean()
        )
        
        # Drop intermediate columns
        df.drop(['high_low', 'high_close', 'low_close', 'true_range'], axis=1, inplace=True)
        
        print("✓ ATR created")
        
        return df
    
    def create_liquidity_ratio(self, df):
        """
        Create liquidity ratio
        Measures trading activity relative to market size
        
        Formula: Volume / Market Cap
        Higher ratio = more liquid market
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with liquidity_ratio column
        """
        print("\n💧 Creating liquidity ratio...")
        
        # Add small constant to avoid division by zero
        df['liquidity_ratio'] = df['volume'] / (df['marketCap'] + 1)
        
        print("✓ Liquidity ratio created")
        
        return df
    
    def create_momentum_features(self, df):
        """
        Create momentum-based features
        
        Features:
        - rsi_14: 14-day Relative Strength Index
        - momentum_7: 7-day price momentum
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with momentum features
        """
        print("\n⚡ Creating momentum features...")
        
        # 7-day momentum (current price / price 7 days ago)
        df['momentum_7'] = df.groupby('crypto_name')['close'].transform(
            lambda x: x / x.shift(7)
        )
        print("✓ 7-day momentum created")
        
        return df
    
    def create_all_features(self, df):
        """
        Execute complete feature engineering pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with all engineered features
        """
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        
        initial_rows = len(df)
        
        # Create all features
        df = self.create_log_returns(df)
        df = self.create_rolling_volatility(df)
        df = self.create_moving_averages(df)
        df = self.create_bollinger_bands(df)
        df = self.create_atr(df)
        df = self.create_liquidity_ratio(df)
        df = self.create_momentum_features(df)
        
        # Remove NaN values created by rolling windows
        print("\n🧹 Removing NaN values from rolling calculations...")
        df = df.dropna()
        rows_removed = initial_rows - len(df)
        
        print(f"\n" + "=" * 60)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 60)
        print(f"✓ Initial rows: {initial_rows:,}")
        print(f"✓ Rows removed (NaN): {rows_removed:,}")
        print(f"✓ Final rows: {len(df):,}")
        print(f"✓ Total columns: {df.shape[1]}")
        print(f"\n✓ Features created successfully!")
        
        return df
    
    def display_feature_info(self, df):
        """
        Display information about created features
        
        Args:
            df (pd.DataFrame): Dataframe with features
        """
        print("\n" + "=" * 60)
        print("FEATURE INFORMATION")
        print("=" * 60)
        
        feature_cols = [
            'log_return', 'volatility_7d', 'volatility_14d',
            'ma_7', 'ma_14', 'ma_30', 'bb_width', 'atr',
            'liquidity_ratio', 'momentum_7'
        ]
        
        print("\n📊 Feature Statistics:")
        print(df[feature_cols].describe())
        
        print("\n🎯 Target Variable (volatility_7d):")
        print(f"  Mean: {df['volatility_7d'].mean():.6f}")
        print(f"  Std: {df['volatility_7d'].std():.6f}")
        print(f"  Min: {df['volatility_7d'].min():.6f}")
        print(f"  Max: {df['volatility_7d'].max():.6f}")
    
    def save_features(self, df, filepath):
        """
        Save feature-engineered data
        
        Args:
            df (pd.DataFrame): Dataframe with features
            filepath (str): Output file path
        """
        print("\n" + "=" * 60)
        print("SAVING FEATURES")
        print("=" * 60)
        
        df.to_csv(filepath, index=False)
        print(f"✓ Features saved to: {filepath}")
        print(f"  Size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Main execution
if __name__ == "__main__":
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Load cleaned data
    df = engineer.load_data('data/processed/cleaned_data.csv')
    
    # Create all features
    df = engineer.create_all_features(df)
    
    # Display feature information
    engineer.display_feature_info(df)
    
    # Save features
    engineer.save_features(df, 'data/processed/features.csv')
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("=" * 60)