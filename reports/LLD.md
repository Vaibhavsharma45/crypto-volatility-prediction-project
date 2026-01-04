# Low Level Design (LLD)
## Cryptocurrency Volatility Prediction System

---

## 1. Module-wise Breakdown

### 1.1 Module Overview

| Module | File | Purpose | Dependencies |
|--------|------|---------|--------------|
| Preprocessing | `preprocessing.py` | Data cleaning and validation | pandas, numpy, sklearn |
| Feature Engineering | `feature_engineering.py` | Feature creation | pandas, numpy |
| Model Training | `train_model.py` | ML model training | sklearn, pickle |
| Model Evaluation | `evaluate_model.py` | Performance assessment | sklearn, matplotlib |
| Deployment | `app.py` | Web interface | streamlit, plotly |

---

## 2. Data Preprocessing Module

### 2.1 Class: `DataPreprocessor`

```python
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
```

### 2.2 Methods

#### `load_data(filepath: str) -> pd.DataFrame`
**Purpose:** Load raw CSV data  
**Input:** File path to CSV  
**Output:** pandas DataFrame  
**Process:**
1. Read CSV using pandas
2. Display dataset dimensions
3. Calculate memory usage
4. Return DataFrame

#### `handle_missing_values(df: pd.DataFrame) -> pd.DataFrame`
**Purpose:** Handle missing values in dataset  
**Input:** DataFrame with potential missing values  
**Output:** DataFrame with handled missing values  
**Process:**
1. Identify missing values by column
2. Drop rows with missing OHLC or volume (critical data)
3. Fill marketCap with group median (by cryptocurrency)
4. Report missing value statistics

**Rationale:**
- OHLC and volume are essential for technical analysis
- MarketCap can be imputed using group statistics
- Preserves maximum data while maintaining quality

#### `ensure_data_consistency(df: pd.DataFrame) -> pd.DataFrame`
**Purpose:** Validate and clean data  
**Input:** DataFrame with potential inconsistencies  
**Output:** Validated DataFrame  
**Process:**
1. Remove duplicate rows
2. Ensure all numeric columns > 0
3. Validate high >= low (price consistency)
4. Validate close within [low, high] range
5. Sort by cryptocurrency and date
6. Reset index

**Business Rules:**
- Prices cannot be negative
- High price must be >= low price
- Close price must be within daily range
- Data must be chronologically ordered

#### `add_basic_features(df: pd.DataFrame) -> pd.DataFrame`
**Purpose:** Add simple calculated features  
**Input:** Clean DataFrame  
**Output:** DataFrame with additional features  
**Features Added:**
- `price_change` = close - open
- `price_change_pct` = ((close - open) / open) * 100
- `daily_range` = high - low

---

## 3. Feature Engineering Module

### 3.1 Class: `FeatureEngineer`

```python
class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
```

### 3.2 Feature Creation Methods

#### `create_log_returns(df: pd.DataFrame) -> pd.DataFrame`
**Formula:** `log_return = log(price_t / price_t-1)`  
**Purpose:** Calculate logarithmic returns  
**Advantages:**
- Time-additive
- Symmetric
- Normally distributed (more suitable for ML)

**Implementation:**
```python
df['log_return'] = df.groupby('crypto_name')['close'].transform(
    lambda x: np.log(x / x.shift(1))
)
```

#### `create_rolling_volatility(df: pd.DataFrame) -> pd.DataFrame`
**Features:**
- `volatility_7d` - TARGET VARIABLE
- `volatility_14d` - Feature for prediction

**Formula:** `volatility = std(log_returns) over window`

**Implementation:**
```python
df['volatility_7d'] = df.groupby('crypto_name')['log_return'].transform(
    lambda x: x.rolling(window=7).std()
)
```

**Window Selection Rationale:**
- 7-day: Short-term volatility (target)
- 14-day: Medium-term volatility (predictor)

#### `create_moving_averages(df: pd.DataFrame) -> pd.DataFrame`
**Features:** MA7, MA14, MA30  
**Purpose:** Capture price trends

**Interpretation:**
- MA7: Short-term trend
- MA14: Medium-term trend
- MA30: Long-term trend

**Cross-overs indicate trend changes**

#### `create_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame`
**Formula:**
```
Middle Band = 20-day SMA
Upper Band = Middle + (2 × std)
Lower Band = Middle - (2 × std)
BB Width = (Upper - Lower) / Middle
```

**Purpose:** Measure market volatility  
**Interpretation:**
- Wide bands = high volatility
- Narrow bands = low volatility
- Price touching bands = potential reversal

#### `create_atr(df: pd.DataFrame) -> pd.DataFrame`
**Formula:**
```
True Range = max(
    high - low,
    |high - prev_close|,
    |low - prev_close|
)
ATR = 14-day average of True Range
```

**Purpose:** Measure market volatility  
**Advantages:**
- Accounts for gaps
- More robust than simple range
- Popular in trading systems

#### `create_liquidity_ratio(df: pd.DataFrame) -> pd.DataFrame`
**Formula:** `liquidity_ratio = volume / (marketCap + 1)`

**Purpose:** Measure trading activity relative to market size  
**Interpretation:**
- High ratio = liquid market, easier to trade
- Low ratio = illiquid market, price slippage risk

#### `create_momentum_features(df: pd.DataFrame) -> pd.DataFrame`
**Formula:** `momentum_7 = close_t / close_t-7`

**Purpose:** Capture price momentum  
**Interpretation:**
- > 1: Upward momentum
- < 1: Downward momentum
- = 1: No momentum

---

## 4. Model Training Module

### 4.1 Class: `ModelTrainer`

```python
class ModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.best_params = None
```

### 4.2 Training Pipeline Methods

#### `prepare_features(df: pd.DataFrame) -> tuple`
**Purpose:** Select and prepare features for training  
**Output:** (X, y, df_clean)

**Feature Selection:**
```python
features = [
    'open', 'high', 'low', 'close', 'volume', 'marketCap',
    'log_return', 'volatility_14d', 
    'ma_7', 'ma_14', 'ma_30',
    'bb_width', 'atr', 
    'liquidity_ratio', 'momentum_7'
]
target = 'volatility_7d'
```

**Rationale:** Features selected based on financial domain knowledge and predictive power

#### `time_based_split(X, y, df, train_size=0.8) -> tuple`
**Purpose:** Split data chronologically  
**Split Ratio:** 80% train, 20% test

**Critical Difference from Random Split:**
- Time series data has temporal dependency
- Random split would leak future information
- Chronological split simulates real-world prediction

**Process:**
1. Sort by date
2. Calculate split index (80% mark)
3. Split: train = [0:split], test = [split:end]
4. Maintain temporal order

#### `scale_features(X_train, X_test) -> tuple`
**Purpose:** Normalize features to same scale  
**Method:** StandardScaler (zero mean, unit variance)

**Formula:** `z = (x - μ) / σ`

**Why Scale?**
- Random Forest doesn't require scaling
- BUT: Improves convergence speed
- Reduces numerical instability
- Makes feature comparison fair

**Critical:** Fit on train, transform both

```python
X_train_scaled = scaler.fit_transform(X_train)  # Fit + transform
X_test_scaled = scaler.transform(X_test)         # Only transform
```

#### `hyperparameter_tuning(X_train, y_train) -> RandomForestRegressor`
**Purpose:** Find optimal model parameters

**Parameter Grid:**
```python
param_grid = {
    'n_estimators': [100, 200, 300],        # Number of trees
    'max_depth': [10, 20, 30, None],        # Tree depth
    'min_samples_split': [2, 5, 10],        # Min samples to split
    'min_samples_leaf': [1, 2, 4],          # Min samples in leaf
    'max_features': ['sqrt', 'log2']        # Features per split
}
```

**Search Strategy:**
- Method: GridSearchCV
- Cross-validation: 3-fold
- Scoring: Negative MSE
- Total combinations: 3 × 4 × 3 × 3 × 2 = 216

**Process:**
1. Define parameter grid
2. Initialize GridSearchCV
3. Fit on training data
4. Select best parameters
5. Return best model

---

## 5. Model Evaluation Module

### 5.1 Class: `ModelEvaluator`

```python
class ModelEvaluator:
    def __init__(self):
        self.model = None
        self.metrics = {}
```

### 5.2 Evaluation Methods

#### `calculate_metrics(y_test, y_pred) -> dict`
**Metrics Calculated:**

1. **RMSE (Root Mean Squared Error)**
   - Formula: `√(Σ(y_pred - y_actual)² / n)`
   - Unit: Same as target variable
   - Interpretation: Average prediction error
   - Lower is better

2. **MAE (Mean Absolute Error)**
   - Formula: `Σ|y_pred - y_actual| / n`
   - Unit: Same as target variable
   - Interpretation: Average absolute error
   - Lower is better
   - Less sensitive to outliers than RMSE

3. **R² Score (Coefficient of Determination)**
   - Formula: `1 - (SS_res / SS_tot)`
   - Range: (-∞, 1]
   - Interpretation: Proportion of variance explained
   - Higher is better
   - 1.0 = perfect predictions
   - 0.0 = model no better than mean

4. **MAPE (Mean Absolute Percentage Error)**
   - Formula: `(100/n) × Σ|y_pred - y_actual| / y_actual`
   - Unit: Percentage
   - Interpretation: Average % error
   - Lower is better

#### `plot_predictions(y_test, y_pred, save_path)`
**Visualizations Generated:**

1. **Actual vs Predicted Scatter**
   - X-axis: Actual values
   - Y-axis: Predicted values
   - Diagonal line: Perfect prediction
   - Interpretation: Points near line = good predictions

2. **Residual Plot**
   - X-axis: Predicted values
   - Y-axis: Residuals (actual - predicted)
   - Horizontal line at y=0
   - Interpretation: Random scatter = good model

3. **Error Distribution**
   - Histogram of residuals
   - Should be approximately normal
   - Centered at zero
   - Interpretation: Symmetric = unbiased model

4. **Time Series Comparison**
   - X-axis: Sample index
   - Y-axis: Volatility
   - Two lines: Actual vs Predicted
   - Interpretation: Visual fit quality

#### `plot_feature_importance(save_path)`
**Purpose:** Identify most influential features

**Process:**
1. Extract feature importances from Random Forest
2. Sort by importance
3. Create horizontal bar chart
4. Display top 15 features

**Interpretation:**
- Higher value = more important
- Helps understand model decisions
- Guides feature engineering improvements

---

## 6. Deployment Module

### 6.1 Streamlit App Structure

```python
def main():
    # 1. Header and configuration
    # 2. Load model
    # 3. Sidebar: File upload and crypto selection
    # 4. Main area: Market overview
    # 5. Prediction section
    # 6. Visualization
    # 7. Download results
```

### 6.2 Key Functions

#### `load_model() -> dict`
**Caching:** `@st.cache_resource`  
**Purpose:** Load model once, reuse across sessions  
**Returns:** Model package with model, scaler, features

#### `create_features(df) -> pd.DataFrame`
**Purpose:** Recreate features for user-uploaded data  
**Must Match:** Training feature pipeline exactly  
**Process:** Apply same transformations as training

#### Prediction Workflow
```
1. User uploads CSV
2. Validate columns
3. Select cryptocurrency
4. Display market overview
5. User clicks "Predict"
6. Create features
7. Scale features
8. Generate predictions
9. Display results
10. Enable download
```

---

## 7. Data Flow Diagrams

### 7.1 Training Data Flow

```
Raw CSV
   ↓
[Load Data]
   ↓
[Handle Missing Values]
   ↓
[Ensure Consistency]
   ↓
Cleaned Data
   ↓
[Create Log Returns]
   ↓
[Create Volatility]
   ↓
[Create Moving Averages]
   ↓
[Create Bollinger Bands]
   ↓
[Create ATR]
   ↓
[Create Liquidity Ratio]
   ↓
[Create Momentum]
   ↓
Feature Dataset
   ↓
[Time-based Split]
   ↓
Train Set | Test Set
   ↓           ↓
[Scale]    [Scale]
   ↓           ↓
[Train RF]  [Evaluate]
   ↓           ↓
Trained    Metrics
Model
```

### 7.2 Prediction Data Flow

```
User Upload
   ↓
[Validate Format]
   ↓
[Select Crypto]
   ↓
[Create Features]
   ↓
Feature Dataset
   ↓
[Scale with Saved Scaler]
   ↓
Scaled Features
   ↓
[Predict with Saved Model]
   ↓
Predictions
   ↓
[Visualize]
   ↓
Results Display
```

---

## 8. Error Handling

### 8.1 Common Errors and Solutions

| Error Type | Cause | Solution |
|------------|-------|----------|
| FileNotFoundError | Missing data file | Check file path, create directories |
| KeyError | Missing column | Validate CSV columns before processing |
| ValueError | Invalid data type | Convert to appropriate type, handle exceptions |
| MemoryError | Dataset too large | Implement chunking or sampling |
| ModelNotFoundError | Model not trained | Run training pipeline first |

### 8.2 Error Handling Pattern

```python
try:
    # Operation
    result = risky_operation()
except SpecificError as e:
    # Log error
    logger.error(f"Error: {e}")
    # User-friendly message
    print("❌ Operation failed. Please check...")
    # Graceful degradation
    return default_value
```

---

## 9. Performance Considerations

### 9.1 Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Data Loading | O(n) | n = rows |
| Feature Creation | O(n × w) | w = window size |
| Model Training | O(n × m × t) | m = features, t = trees |
| Prediction | O(m × t) | Fast |

### 9.2 Space Complexity

| Component | Size | Optimization |
|-----------|------|--------------|
| Raw Data | ~50 MB | Use chunking for larger datasets |
| Features | ~100 MB | Drop unnecessary columns |
| Model | ~10-50 MB | Depends on tree count |
| Predictions | ~1 MB | Minimal |

### 9.3 Optimization Strategies

1. **Vectorization:** Use numpy/pandas operations
2. **Groupby Operations:** Efficient for cryptocurrency grouping
3. **Memory Management:** Delete unused dataframes
4. **Caching:** Cache model loading in Streamlit
5. **Parallel Processing:** Use n_jobs=-1 in Random Forest

---

## 10. Testing Strategy

### 10.1 Unit Tests
- Test each function independently
- Mock external dependencies
- Verify edge cases

### 10.2 Integration Tests
- Test module interactions
- Verify data flow
- Check end-to-end pipeline

### 10.3 Validation Tests
- Cross-validation during training
- Holdout test set evaluation
- Performance monitoring

---

## 11. Conclusion

This Low Level Design provides detailed implementation specifications for the Cryptocurrency Volatility Prediction System. Each module is designed to be modular, testable, and maintainable, following software engineering best practices.

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Author:** PW Skills Student