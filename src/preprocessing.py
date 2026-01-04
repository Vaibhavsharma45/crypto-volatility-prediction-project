"""
Data Preprocessing Module
Handles missing values, data cleaning, and normalization
Author: PW Skills Student
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Preprocesses cryptocurrency market data
    Handles missing values, ensures consistency, and normalizes features
    """
    
    def __init__(self):
        """Initialize the preprocessor with a StandardScaler"""
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """
        Load dataset from CSV file
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded successfully")
        print(f"  - Rows: {df.shape[0]:,}")
        print(f"  - Columns: {df.shape[1]}")
        print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        - Drop rows with missing critical price/volume data
        - Fill marketCap with group median
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        print("\n" + "=" * 60)
        print("HANDLING MISSING VALUES")
        print("=" * 60)
        
        # Display missing value summary
        missing_before = df.isnull().sum()
        print("\nMissing values before cleaning:")
        print(missing_before[missing_before > 0])
        print(f"Total missing: {df.isnull().sum().sum():,}")
        
        # Drop rows with missing critical values (OHLC and volume)
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        df = df.dropna(subset=critical_cols)
        print(f"\n✓ Dropped rows with missing critical values: {critical_cols}")
        
        # Fill marketCap with group median (by cryptocurrency)
        if 'marketCap' in df.columns and df['marketCap'].isnull().any():
            df['marketCap'] = df.groupby('crypto_name')['marketCap'].transform(
                lambda x: x.fillna(x.median())
            )
            print("✓ Filled missing marketCap with group median")
        
        # Display final missing values
        missing_after = df.isnull().sum()
        print("\nMissing values after cleaning:")
        print(f"Total missing: {df.isnull().sum().sum()}")
        
        return df
    
    def ensure_data_consistency(self, df):
        """
        Ensure data quality and consistency
        - Remove duplicates
        - Ensure positive values for all numerical columns
        - Ensure high >= low (price consistency)
        - Sort by cryptocurrency and date
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned and consistent dataframe
        """
        print("\n" + "=" * 60)
        print("ENSURING DATA CONSISTENCY")
        print("=" * 60)
        
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        print(f"✓ Removed {duplicates_removed:,} duplicate rows")
        
        # Ensure positive values for price and volume columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'marketCap']
        rows_before = len(df)
        for col in numeric_cols:
            if col in df.columns:
                df = df[df[col] > 0]
        rows_removed = rows_before - len(df)
        print(f"✓ Removed {rows_removed:,} rows with non-positive values")
        
        # Ensure price consistency (high >= low)
        rows_before = len(df)
        df = df[df['high'] >= df['low']]
        inconsistent_removed = rows_before - len(df)
        print(f"✓ Removed {inconsistent_removed:,} rows where high < low")
        
        # Ensure close price is within high-low range
        df = df[(df['close'] >= df['low']) & (df['close'] <= df['high'])]
        
        # Convert date to datetime and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['crypto_name', 'date']).reset_index(drop=True)
        print("✓ Data sorted by cryptocurrency and date")
        
        print(f"\n✓ Final dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"✓ Total rows removed: {initial_rows - len(df):,}")
        
        return df
    
    def add_basic_features(self, df):
        """
        Add basic calculated features
        - Price change
        - Price change percentage
        - Daily range
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        print("\n" + "=" * 60)
        print("ADDING BASIC FEATURES")
        print("=" * 60)
        
        # Price change
        df['price_change'] = df['close'] - df['open']
        print("✓ Added price_change")
        
        # Price change percentage
        df['price_change_pct'] = ((df['close'] - df['open']) / df['open']) * 100
        print("✓ Added price_change_pct")
        
        # Daily range (high - low)
        df['daily_range'] = df['high'] - df['low']
        print("✓ Added daily_range")
        
        return df
    
    def normalize_features(self, df, features):
        """
        Normalize numerical features using StandardScaler
        
        Args:
            df (pd.DataFrame): Input dataframe
            features (list): List of features to normalize
            
        Returns:
            pd.DataFrame: Dataframe with normalized features
        """
        print("\n" + "=" * 60)
        print("NORMALIZING FEATURES")
        print("=" * 60)
        
        df_scaled = df.copy()
        df_scaled[features] = self.scaler.fit_transform(df[features])
        print(f"✓ Normalized {len(features)} features")
        print(f"  Features: {', '.join(features[:5])}...")
        
        return df_scaled
    
    def get_data_summary(self, df):
        """
        Display comprehensive data summary
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        
        print("\nCryptocurrency distribution:")
        print(df['crypto_name'].value_counts())
        
        print("\nDate range:")
        print(f"  Start: {df['date'].min()}")
        print(f"  End: {df['date'].max()}")
        print(f"  Duration: {(df['date'].max() - df['date'].min()).days} days")
        
        print("\nNumerical features summary:")
        print(df[['open', 'high', 'low', 'close', 'volume', 'marketCap']].describe())
    
    def save_processed_data(self, df, filepath):
        """
        Save processed data to CSV
        
        Args:
            df (pd.DataFrame): Processed dataframe
            filepath (str): Output file path
        """
        print("\n" + "=" * 60)
        print("SAVING PROCESSED DATA")
        print("=" * 60)
        
        df.to_csv(filepath, index=False)
        print(f"✓ Processed data saved to: {filepath}")
        print(f"  Size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Main execution
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load raw data
    df = preprocessor.load_data('data/raw/dataset.csv')
    
    # Preprocessing pipeline
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.ensure_data_consistency(df)
    df = preprocessor.add_basic_features(df)
    
    # Display summary
    preprocessor.get_data_summary(df)
    
    # Save processed data
    preprocessor.save_processed_data(df, 'data/processed/cleaned_data.csv')
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)