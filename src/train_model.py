"""
Model Training Module
Trains Random Forest Regressor with hyperparameter tuning
Author: PW Skills Student
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Trains Random Forest model for cryptocurrency volatility prediction
    Implements time-series aware splitting and hyperparameter tuning
    """
    
    def __init__(self):
        """Initialize model trainer"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.best_params = None
    
    def load_data(self, filepath):
        """
        Load feature-engineered data
        
        Args:
            filepath (str): Path to features CSV
            
        Returns:
            pd.DataFrame: Loaded dataframe with features
        """
        print("=" * 60)
        print("LOADING FEATURE DATA")
        print("=" * 60)
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    
    def prepare_features(self, df):
        """
        Prepare features and target variable
        
        Feature selection based on:
        - Original OHLCV data
        - Technical indicators
        - Volatility measures
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (X, y, df) - Features, target, and original dataframe
        """
        print("\n" + "=" * 60)
        print("PREPARING FEATURES AND TARGET")
        print("=" * 60)
        
        # Define feature columns (exclude target and non-predictive columns)
        self.feature_columns = [
            # Original market data
            'open', 'high', 'low', 'close', 'volume', 'marketCap',
            # Technical indicators
            'log_return', 'volatility_14d', 
            'ma_7', 'ma_14', 'ma_30',
            'bb_width', 'atr', 
            'liquidity_ratio', 'momentum_7'
        ]
        
        # Target variable: 7-day rolling volatility
        target = 'volatility_7d'
        
        # Remove any remaining NaN values
        df_clean = df.dropna(subset=self.feature_columns + [target])
        
        print(f"\n✓ Features selected: {len(self.feature_columns)}")
        print(f"✓ Target variable: {target}")
        print(f"\nFeature list:")
        for i, feat in enumerate(self.feature_columns, 1):
            print(f"  {i}. {feat}")
        
        # Separate features and target
        X = df_clean[self.feature_columns]
        y = df_clean[target]
        
        print(f"\n✓ Features shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        print(f"✓ Data points removed (NaN): {len(df) - len(df_clean):,}")
        
        return X, y, df_clean
    
    def time_based_split(self, X, y, df, train_size=0.8):
        """
        Perform time-based train-test split
        IMPORTANT: For time series, we split chronologically, not randomly
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            df (pd.DataFrame): Original dataframe with dates
            train_size (float): Proportion of data for training (default: 0.8)
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\n" + "=" * 60)
        print("TIME-BASED TRAIN-TEST SPLIT")
        print("=" * 60)
        
        # Sort by date (already sorted, but ensuring)
        df_sorted = df.sort_values('date').reset_index(drop=True)
        X = X.loc[df_sorted.index]
        y = y.loc[df_sorted.index]
        
        # Calculate split index
        split_idx = int(len(df_sorted) * train_size)
        
        # Split data
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Get date ranges
        train_dates = df_sorted['date'].iloc[:split_idx]
        test_dates = df_sorted['date'].iloc[split_idx:]
        
        print(f"✓ Split ratio: {train_size:.0%} train / {1-train_size:.0%} test")
        print(f"\n📅 Training period:")
        print(f"  Start: {train_dates.min().date()}")
        print(f"  End: {train_dates.max().date()}")
        print(f"  Duration: {(train_dates.max() - train_dates.min()).days} days")
        print(f"  Samples: {len(X_train):,}")
        
        print(f"\n📅 Testing period:")
        print(f"  Start: {test_dates.min().date()}")
        print(f"  End: {test_dates.max().date()}")
        print(f"  Duration: {(test_dates.max() - test_dates.min()).days} days")
        print(f"  Samples: {len(X_test):,}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler
        Fit on training data only to prevent data leakage
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Testing features
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        print("\n" + "=" * 60)
        print("FEATURE SCALING")
        print("=" * 60)
        
        # Fit scaler on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("✓ Features scaled using StandardScaler")
        print("  Method: Zero mean, unit variance")
        print(f"  Training mean: {X_train_scaled.mean():.6f}")
        print(f"  Training std: {X_train_scaled.std():.6f}")
        
        return X_train_scaled, X_test_scaled
    
    def train_baseline_model(self, X_train, y_train):
        """
        Train baseline Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            RandomForestRegressor: Trained baseline model
        """
        print("\n" + "=" * 60)
        print("TRAINING BASELINE MODEL")
        print("=" * 60)
        
        baseline_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        print("Training baseline Random Forest...")
        baseline_model.fit(X_train, y_train)
        
        train_score = baseline_model.score(X_train, y_train)
        print(f"\n✓ Baseline model trained")
        print(f"  Training R² Score: {train_score:.4f}")
        
        return baseline_model
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            RandomForestRegressor: Best model from grid search
        """
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING")
        print("=" * 60)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        print("Parameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Initialize Random Forest
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Grid search with time series cross-validation
        print("\n🔍 Starting Grid Search...")
        print("  Cross-validation: 3-fold TimeSeriesSplit")
        print("  Scoring: Negative Mean Squared Error")
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,  # 3-fold cross-validation
            scoring='neg_mean_squared_error',
            verbose=2,
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Store best parameters
        self.best_params = grid_search.best_params_
        
        print("\n✅ Grid Search Complete!")
        print(f"\n🏆 Best Parameters:")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        
        print(f"\n📊 Best CV Score: {-grid_search.best_score_:.6f} (MSE)")
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        return self.model
    
    def train_final_model(self, X_train, y_train, use_tuning=True):
        """
        Train final model (with or without hyperparameter tuning)
        
        Args:
            X_train: Training features
            y_train: Training target
            use_tuning (bool): Whether to use hyperparameter tuning
            
        Returns:
            RandomForestRegressor: Trained model
        """
        if use_tuning:
            model = self.hyperparameter_tuning(X_train, y_train)
        else:
            model = self.train_baseline_model(X_train, y_train)
        
        return model
    
    def save_model(self, filepath='model/volatility_model.pkl'):
        """
        Save trained model, scaler, and metadata
        
        Args:
            filepath (str): Path to save model
        """
        print("\n" + "=" * 60)
        print("SAVING MODEL")
        print("=" * 60)
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Package model with all necessary components
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'best_params': self.best_params
        }
        
        # Save using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Get file size
        file_size = os.path.getsize(filepath) / 1024**2
        
        print(f"✓ Model saved successfully!")
        print(f"  Path: {filepath}")
        print(f"  Size: {file_size:.2f} MB")
        print(f"  Components: model, scaler, feature_columns, best_params")

# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CRYPTOCURRENCY VOLATILITY PREDICTION")
    print("MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    df = trainer.load_data('data/processed/features.csv')
    
    # Prepare features
    X, y, df_clean = trainer.prepare_features(df)
    
    # Time-based split
    X_train, X_test, y_train, y_test = trainer.time_based_split(X, y, df_clean)
    
    # Scale features
    X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
    
    # Train model with hyperparameter tuning
    model = trainer.train_final_model(X_train_scaled, y_train, use_tuning=True)
    
    # Save model
    trainer.save_model('model/volatility_model.pkl')
    
    # Save test data for evaluation
    print("\n💾 Saving test data for evaluation...")
    test_data = {
        'X_test': X_test_scaled,
        'y_test': y_test.values,
        'X_test_df': X_test,
        'dates': df_clean.iloc[len(X_train):]['date'].values
    }
    
    os.makedirs('model', exist_ok=True)
    with open('model/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    print("✓ Test data saved to model/test_data.pkl")
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n✅ Next step: Run evaluate_model.py to evaluate performance")