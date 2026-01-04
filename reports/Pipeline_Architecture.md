# Pipeline Architecture
## Cryptocurrency Volatility Prediction ML Pipeline

---

## 1. Pipeline Overview

The ML pipeline consists of five major stages that transform raw cryptocurrency market data into actionable volatility predictions. Each stage is designed to be independent, testable, and maintainable.

```
┌──────────────────────────────────────────────────────────────────┐
│                     ML PIPELINE OVERVIEW                         │
│                                                                  │
│  Raw Data → Preprocessing → Feature Engineering → Model →       │
│  Evaluation → Deployment                                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Pipeline Architecture

### 2.1 ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA INGESTION                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐      ┌──────────────────┐                        │
│  │ dataset.csv │─────>│  Load CSV Data   │                        │
│  └─────────────┘      └────────┬─────────┘                        │
│                                 │                                  │
│                                 ▼                                  │
│                      ┌─────────────────────┐                      │
│                      │  pandas DataFrame   │                      │
│                      │  72,946 rows        │                      │
│                      │  10 columns         │                      │
│                      └──────────┬──────────┘                      │
└──────────────────────────────────┼──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: DATA PREPROCESSING                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────┐         ┌─────────────────────┐           │
│  │ Missing Value      │────────>│ Data Consistency    │           │
│  │ Handling           │         │ Validation          │           │
│  │                    │         │                     │           │
│  │ • Drop critical NaN│         │ • Remove duplicates │           │
│  │ • Fill marketCap   │         │ • Ensure positive   │           │
│  └────────┬───────────┘         │ • Validate prices   │           │
│           │                     │ • Sort by date      │           │
│           │                     └──────────┬──────────┘           │
│           │                                │                      │
│           └────────────────────────────────┘                      │
│                                 │                                  │
│                                 ▼                                  │
│                    ┌────────────────────────┐                     │
│                    │  Cleaned Dataset       │                     │
│                    │  ~70,000 rows          │                     │
│                    │  data/processed/       │                     │
│                    │  cleaned_data.csv      │                     │
│                    └───────────┬────────────┘                     │
└──────────────────────────────────┼──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: FEATURE ENGINEERING                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐        │
│  │ Log Returns │──>│ Volatility   │──>│ Moving Averages │        │
│  │             │   │ (7d, 14d)    │   │ (MA7,14,30)     │        │
│  └─────────────┘   └──────────────┘   └────────┬────────┘        │
│                                                 │                  │
│  ┌─────────────┐   ┌──────────────┐          │                  │
│  │ Bollinger   │──>│     ATR      │<──────────┘                  │
│  │ Bands       │   │              │                              │
│  └─────────────┘   └──────┬───────┘                              │
│                            │                                      │
│  ┌─────────────┐   ┌──────┴───────┐                              │
│  │ Liquidity   │──>│  Momentum    │                              │
│  │ Ratio       │   │   Features   │                              │
│  └─────────────┘   └──────┬───────┘                              │
│                            │                                      │
│                            ▼                                      │
│              ┌──────────────────────────┐                         │
│              │  Feature Dataset         │                         │
│              │  14 Features + Target    │                         │
│              │  data/processed/         │                         │
│              │  features.csv            │                         │
│              └────────────┬─────────────┘                         │
└──────────────────────────────┼──────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4: MODEL TRAINING & EVALUATION                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────┐                                         │
│  │  Time-based Split     │                                         │
│  │  • Train: 80%         │                                         │
│  │  • Test: 20%          │                                         │
│  └──────────┬────────────┘                                         │
│             │                                                      │
│      ┌──────┴──────┐                                              │
│      ▼             ▼                                              │
│  ┌────────┐   ┌────────┐                                          │
│  │ Train  │   │  Test  │                                          │
│  │ Set    │   │  Set   │                                          │
│  └────┬───┘   └───┬────┘                                          │
│       │           │                                                │
│       ▼           │                                                │
│  ┌────────────────┐                                               │
│  │ Feature Scaling│                                               │
│  │ StandardScaler │                                               │
│  └────────┬───────┘                                               │
│           │                                                        │
│           ▼                                                        │
│  ┌────────────────────────┐                                       │
│  │ Hyperparameter Tuning  │                                       │
│  │                        │                                       │
│  │ GridSearchCV:          │                                       │
│  │ • n_estimators         │                                       │
│  │ • max_depth            │                                       │
│  │ • min_samples_split    │                                       │
│  │ • min_samples_leaf     │                                       │
│  │ • max_features         │                                       │
│  │                        │                                       │
│  │ 3-Fold CV              │                                       │
│  └───────────┬────────────┘                                       │
│              │                                                     │
│              ▼                                                     │
│  ┌─────────────────────────┐                                      │
│  │ Train Random Forest     │                                      │
│  │ with Best Parameters    │                                      │
│  └───────────┬─────────────┘                                      │
│              │                                                     │
│              ▼                                                     │
│  ┌─────────────────────────┐                                      │
│  │  Make Predictions       │<────────── Test Set                  │
│  └───────────┬─────────────┘                                      │
│              │                                                     │
│              ▼                                                     │
│  ┌─────────────────────────┐                                      │
│  │  Calculate Metrics      │                                      │
│  │  • RMSE                 │                                      │
│  │  • MAE                  │                                      │
│  │  • R² Score             │                                      │
│  │  • MAPE                 │                                      │
│  └───────────┬─────────────┘                                      │
│              │                                                     │
│              ▼                                                     │
│  ┌─────────────────────────┐                                      │
│  │  Save Model             │                                      │
│  │  model/                 │                                      │
│  │  volatility_model.pkl   │                                      │
│  └─────────────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 5: DEPLOYMENT                                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐                                               │
│  │ Streamlit App   │                                               │
│  │ app.py          │                                               │
│  └────────┬────────┘                                               │
│           │                                                        │
│           ▼                                                        │
│  ┌─────────────────┐      ┌──────────────────┐                   │
│  │ User Upload CSV │─────>│ Validate Format  │                   │
│  └─────────────────┘      └────────┬─────────┘                   │
│                                     │                              │
│                                     ▼                              │
│                          ┌──────────────────┐                     │
│                          │ Select Crypto    │                     │
│                          └────────┬─────────┘                     │
│                                   │                                │
│                                   ▼                                │
│                      ┌────────────────────────┐                   │
│                      │ Feature Engineering    │                   │
│                      │ (Same as Training)     │                   │
│                      └──────────┬─────────────┘                   │
│                                 │                                  │
│                                 ▼                                  │
│                      ┌────────────────────────┐                   │
│                      │ Scale Features         │                   │
│                      │ (Use Saved Scaler)     │                   │
│                      └──────────┬─────────────┘                   │
│                                 │                                  │
│                                 ▼                                  │
│                      ┌────────────────────────┐                   │
│                      │ Predict with Model     │                   │
│                      └──────────┬─────────────┘                   │
│                                 │                                  │
│                                 ▼                                  │
│                      ┌────────────────────────┐                   │
│                      │ Visualize Results      │                   │
│                      │ • Time series plot     │                   │
│                      │ • Distribution         │                   │
│                      │ • Statistics           │                   │
│                      └──────────┬─────────────┘                   │
│                                 │                                  │
│                                 ▼                                  │
│                      ┌────────────────────────┐                   │
│                      │ Download Predictions   │                   │
│                      └────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage-by-Stage Breakdown

### Stage 1: Data Ingestion
**Input:** `dataset.csv`  
**Output:** pandas DataFrame  
**Duration:** ~2 seconds

**Operations:**
1. Read CSV file
2. Convert date column to datetime
3. Display dataset information
4. Validate column existence

### Stage 2: Data Preprocessing
**Input:** Raw DataFrame  
**Output:** `data/processed/cleaned_data.csv`  
**Duration:** ~5 seconds

**Operations:**
1. Handle missing values (~1%)
2. Remove duplicates (~0.5%)
3. Validate data consistency
4. Filter invalid rows (~2%)
5. Sort by date and cryptocurrency

**Key Decisions:**
- Drop rows with missing OHLC (essential data)
- Impute marketCap with group median
- Remove negative values (data errors)
- Ensure high >= low (business logic)

### Stage 3: Feature Engineering
**Input:** `cleaned_data.csv`  
**Output:** `data/processed/features.csv`  
**Duration:** ~10 seconds

**Features Created:** 14 features + 1 target

**Feature Categories:**
1. **Returns:** log_return
2. **Volatility:** volatility_7d (target), volatility_14d
3. **Trend:** ma_7, ma_14, ma_30
4. **Volatility Indicators:** bb_width, atr
5. **Volume:** liquidity_ratio
6. **Momentum:** momentum_7

**Window Sizes:**
- 7 days: Short-term patterns
- 14 days: Medium-term patterns
- 20 days: Bollinger Bands standard
- 30 days: Long-term trends

### Stage 4: Model Training & Evaluation
**Input:** `features.csv`  
**Output:** `model/volatility_model.pkl`  
**Duration:** ~5-15 minutes (depending on grid search)

**Sub-stages:**

#### 4.1 Data Splitting
- Method: Time-based (not random)
- Ratio: 80% train, 20% test
- Ensures no future data leakage

#### 4.2 Feature Scaling
- Method: StandardScaler
- Fit on train only
- Transform both train and test

#### 4.3 Hyperparameter Tuning
- Algorithm: GridSearchCV
- CV Strategy: 3-fold
- Scoring: Negative MSE
- Combinations tested: 216

**Parameter Space:**
```
n_estimators: [100, 200, 300]
max_depth: [10, 20, 30, None]
min_samples_split: [2, 5, 10]
min_samples_leaf: [1, 2, 4]
max_features: ['sqrt', 'log2']
```

#### 4.4 Model Training
- Algorithm: Random Forest Regressor
- Parameters: Best from grid search
- Training time: ~2-5 minutes

#### 4.5 Model Evaluation
**Metrics:**
- RMSE: ~0.001-0.005
- MAE: ~0.0008-0.003
- R²: ~0.75-0.90
- MAPE: ~5-15%

### Stage 5: Deployment
**Input:** User CSV + Trained model  
**Output:** Interactive predictions  
**Duration:** Real-time (<2 seconds per prediction)

**User Flow:**
```
Upload CSV → Select Crypto → View Overview → Predict → 
Visualize → Download
```

---

## 4. Data Transformations

### 4.1 Input Schema

```
Original Data (dataset.csv):
├── date (datetime)
├── crypto_name (string)
├── open (float)
├── high (float)
├── low (float)
├── close (float)
├── volume (float)
└── marketCap (float)
```

### 4.2 Intermediate Schema

```
After Preprocessing (cleaned_data.csv):
├── All original columns
├── price_change (float)
├── price_change_pct (float)
└── daily_range (float)
```

### 4.3 Final Schema

```
After Feature Engineering (features.csv):
├── All previous columns
├── log_return (float)
├── volatility_7d (float) ← TARGET
├── volatility_14d (float)
├── ma_7, ma_14, ma_30 (float)
├── bb_width (float)
├── atr (float)
├── liquidity_ratio (float)
└── momentum_7 (float)
```

---

## 5. Pipeline Execution

### 5.1 Training Pipeline Execution Order

```bash
# Step 1: Preprocessing
python src/preprocessing.py
# Output: data/processed/cleaned_data.csv

# Step 2: Feature Engineering  
python src/feature_engineering.py
# Output: data/processed/features.csv

# Step 3: Model Training
python src/train_model.py
# Output: model/volatility_model.pkl
#         model/test_data.pkl

# Step 4: Model Evaluation
python src/evaluate_model.py
# Output: reports/model_evaluation.png
#         reports/feature_importance.png
#         reports/evaluation_metrics.txt
```

### 5.2 Deployment Pipeline

```bash
# Launch Streamlit App
streamlit run app.py
# Opens browser at localhost:8501
```

---

## 6. Pipeline Validation

### 6.1 Data Quality Checks

**Checkpoint 1 - After Preprocessing:**
- ✓ No missing values in critical columns
- ✓ All prices > 0
- ✓ High >= Low
- ✓ Data sorted by date
- ✓ No duplicates

**Checkpoint 2 - After Feature Engineering:**
- ✓ No NaN in features
- ✓ Features within expected ranges
- ✓ Target variable properly created
- ✓ Sufficient data after rolling windows

**Checkpoint 3 - After Training:**
- ✓ Model converged
- ✓ Reasonable evaluation metrics
- ✓ No overfitting (train vs test gap)
- ✓ Feature importances make sense

---

## 7. Pipeline Optimization

### 7.1 Performance Bottlenecks

| Stage | Time | Optimization |
|-------|------|--------------|
| Data Loading | 2s | Use chunking for huge files |
| Preprocessing | 5s | Vectorized operations |
| Feature Engineering | 10s | Parallel group operations |
| Model Training | 5-15min | Reduce grid search space |
| Prediction | <1s | Already optimized |

### 7.2 Memory Usage

| Component | Size | Optimization |
|-----------|------|--------------|
| Raw Data | 50 MB | Stream processing |
| Features | 100 MB | Drop unused columns |
| Model | 20 MB | Reduce n_estimators |
| Predictions | 1 MB | Minimal |

---

## 8. Error Handling in Pipeline

### 8.1 Stage-wise Error Recovery

```
Stage 1 Error → Check file exists, correct format
Stage 2 Error → Validate column names, data types
Stage 3 Error → Check sufficient data for windows
Stage 4 Error → Verify feature consistency
Stage 5 Error → Validate model file exists
```

### 8.2 Failure Points

1. **File Not Found**
   - Check: File path correct
   - Solution: Create directories, verify path

2. **Missing Columns**
   - Check: CSV has required columns
   - Solution: Use correct dataset

3. **Insufficient Data**
   - Check: Need 30+ rows for rolling windows
   - Solution: Use larger dataset

4. **Model Not Trained**
   - Check: .pkl file exists
   - Solution: Run training pipeline

---

## 9. Pipeline Monitoring

### 9.1 Key Metrics to Track

**Data Quality:**
- Missing value percentage
- Outlier count
- Data distribution changes

**Model Performance:**
- RMSE trend over time
- R² score
- Prediction drift

**System Performance:**
- Pipeline execution time
- Memory usage
- API response time

---

## 10. Conclusion

This pipeline architecture provides a robust, scalable framework for cryptocurrency volatility prediction. Each stage is independently testable and can be optimized without affecting others. The time-series aware approach ensures realistic model evaluation and prevents data leakage.

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Author:** PW Skills Student