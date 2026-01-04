"""
Model Evaluation Module
Evaluates model performance using RMSE, MAE, and R² Score
Author: PW Skills Student
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

class ModelEvaluator:
    """
    Evaluates trained Random Forest model performance
    Generates comprehensive metrics and visualizations
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.metrics = {}
    
    def load_model(self, filepath='model/volatility_model.pkl'):
        """
        Load trained model and components
        
        Args:
            filepath (str): Path to saved model
            
        Returns:
            dict: Model package with all components
        """
        print("=" * 60)
        print("LOADING MODEL")
        print("=" * 60)
        
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        self.model = model_package['model']
        self.scaler = model_package['scaler']
        self.feature_columns = model_package['feature_columns']
        
        print("✓ Model loaded successfully!")
        print(f"  Model type: {type(self.model).__name__}")
        print(f"  Features: {len(self.feature_columns)}")
        
        # Display model parameters
        if hasattr(self.model, 'get_params'):
            params = self.model.get_params()
            print(f"\n📋 Model Parameters:")
            print(f"  n_estimators: {params.get('n_estimators', 'N/A')}")
            print(f"  max_depth: {params.get('max_depth', 'N/A')}")
            print(f"  min_samples_split: {params.get('min_samples_split', 'N/A')}")
            print(f"  min_samples_leaf: {params.get('min_samples_leaf', 'N/A')}")
        
        return model_package
    
    def load_test_data(self, filepath='model/test_data.pkl'):
        """
        Load test data saved during training
        
        Args:
            filepath (str): Path to test data
            
        Returns:
            tuple: (X_test, y_test)
        """
        print("\n" + "=" * 60)
        print("LOADING TEST DATA")
        print("=" * 60)
        
        with open(filepath, 'rb') as f:
            test_data = pickle.load(f)
        
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        print(f"✓ Test data loaded")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
        return X_test, y_test
    
    def calculate_metrics(self, y_test, y_pred):
        """
        Calculate comprehensive evaluation metrics
        
        Metrics:
        - RMSE: Root Mean Squared Error (lower is better)
        - MAE: Mean Absolute Error (lower is better)
        - R² Score: Coefficient of determination (higher is better)
        - MAPE: Mean Absolute Percentage Error
        
        Args:
            y_test: Actual values
            y_pred: Predicted values
            
        Returns:
            dict: Dictionary of metrics
        """
        print("\n" + "=" * 60)
        print("CALCULATING METRICS")
        print("=" * 60)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        self.metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2,
            'MAPE': mape
        }
        
        # Display metrics
        print("\n📊 EVALUATION METRICS:")
        print("=" * 60)
        print(f"  Root Mean Squared Error (RMSE): {rmse:.8f}")
        print(f"  Mean Absolute Error (MAE):      {mae:.8f}")
        print(f"  R² Score:                       {r2:.6f}")
        print(f"  MAPE (Mean Absolute % Error):   {mape:.4f}%")
        print("=" * 60)
        
        # Interpretation
        print("\n📖 INTERPRETATION:")
        if r2 > 0.8:
            print("  ✅ Excellent model performance (R² > 0.8)")
        elif r2 > 0.6:
            print("  ✅ Good model performance (R² > 0.6)")
        elif r2 > 0.4:
            print("  ⚠️  Moderate model performance (R² > 0.4)")
        else:
            print("  ❌ Poor model performance (R² < 0.4)")
        
        return self.metrics
    
    def plot_predictions(self, y_test, y_pred, save_path='reports'):
        """
        Create comprehensive prediction visualizations
        
        Args:
            y_test: Actual values
            y_pred: Predicted values
            save_path (str): Directory to save plots
        """
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        # Create reports directory
        os.makedirs(save_path, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation - Cryptocurrency Volatility Prediction', 
                     fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted Scatter Plot
        ax1 = axes[0, 0]
        ax1.scatter(y_test, y_pred, alpha=0.5, s=20, color='blue', label='Predictions')
        ax1.plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Volatility', fontweight='bold')
        ax1.set_ylabel('Predicted Volatility', fontweight='bold')
        ax1.set_title('Actual vs Predicted Volatility')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add R² annotation
        r2 = self.metrics['R2_Score']
        ax1.text(0.05, 0.95, f'R² = {r2:.4f}', 
                transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top')
        
        # 2. Residual Plot
        ax2 = axes[0, 1]
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=20, color='green')
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Volatility', fontweight='bold')
        ax2.set_ylabel('Residuals', fontweight='bold')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Error Distribution
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='purple')
        ax3.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
        ax3.set_xlabel('Prediction Error', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Time Series Comparison (first 500 points)
        ax4 = axes[1, 1]
        n_points = min(500, len(y_test))
        x_range = range(n_points)
        ax4.plot(x_range, y_test[:n_points], label='Actual', linewidth=2, alpha=0.7)
        ax4.plot(x_range, y_pred[:n_points], label='Predicted', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Sample Index', fontweight='bold')
        ax4.set_ylabel('Volatility', fontweight='bold')
        ax4.set_title(f'Time Series Comparison (First {n_points} Points)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_path, 'model_evaluation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Evaluation plot saved: {plot_path}")
        plt.close()
    
    def plot_feature_importance(self, save_path='reports'):
        """
        Plot feature importance from Random Forest
        
        Args:
            save_path (str): Directory to save plot
        """
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create dataframe
        feature_imp = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n📊 Top 10 Most Important Features:")
        print("=" * 60)
        for idx, row in feature_imp.head(10).iterrows():
            print(f"  {row['feature']:20s}: {row['importance']:.6f}")
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot top 15 features
        top_features = feature_imp.head(15)
        bars = plt.barh(top_features['feature'], top_features['importance'])
        
        # Color gradient
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Importance Score', fontweight='bold', fontsize=12)
        plt.ylabel('Feature', fontweight='bold', fontsize=12)
        plt.title('Top 15 Feature Importances - Random Forest Model', 
                 fontweight='bold', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(save_path, 'feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Feature importance plot saved: {plot_path}")
        plt.close()
    
    def generate_evaluation_report(self, save_path='reports'):
        """
        Generate text evaluation report
        
        Args:
            save_path (str): Directory to save report
        """
        print("\n" + "=" * 60)
        print("GENERATING EVALUATION REPORT")
        print("=" * 60)
        
        report_path = os.path.join(save_path, 'evaluation_metrics.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("Cryptocurrency Volatility Prediction\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MODEL INFORMATION:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Model Type: Random Forest Regressor\n")
            f.write(f"Number of Features: {len(self.feature_columns)}\n")
            f.write(f"Target Variable: 7-day Rolling Volatility\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 60 + "\n")
            for metric, value in self.metrics.items():
                if metric == 'MAPE':
                    f.write(f"{metric}: {value:.4f}%\n")
                else:
                    f.write(f"{metric}: {value:.8f}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"✓ Evaluation report saved: {report_path}")
    
    def evaluate(self, X_test=None, y_test=None):
        """
        Complete evaluation pipeline
        
        Args:
            X_test: Test features (optional, will load if not provided)
            y_test: Test target (optional, will load if not provided)
        """
        # Load test data if not provided
        if X_test is None or y_test is None:
            X_test, y_test = self.load_test_data()
        
        # Make predictions
        print("\n🔮 Making predictions on test set...")
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        self.calculate_metrics(y_test, y_pred)
        
        # Generate visualizations
        self.plot_predictions(y_test, y_pred)
        self.plot_feature_importance()
        
        # Generate report
        self.generate_evaluation_report()
        
        return y_pred

# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CRYPTOCURRENCY VOLATILITY PREDICTION")
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load model
    evaluator.load_model('model/volatility_model.pkl')
    
    # Load test data
    X_test, y_test = evaluator.load_test_data('model/test_data.pkl')
    
    # Evaluate model
    y_pred = evaluator.evaluate(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📁 Generated files:")
    print("  - reports/model_evaluation.png")
    print("  - reports/feature_importance.png")
    print("  - reports/evaluation_metrics.txt")
    print("\n✅ Next step: Deploy model using app.py (Streamlit)")