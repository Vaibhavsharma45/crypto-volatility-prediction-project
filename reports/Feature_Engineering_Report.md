
# Feature Engineering Report
## Cryptocurrency Volatility Prediction

Generated: 2026-01-04 10:32:41.408399

## Features Created: 10

### 1. Log Returns
- Formula: ln(Price_t / Price_t-1)
- Purpose: Calculate returns for volatility
- Mean: 0.001006

### 2. Volatility (7-day) - TARGET VARIABLE
- Formula: std(log_returns over 7 days)
- Purpose: Target variable for prediction
- Mean: 0.054645

### 3. Volatility (14-day)
- Formula: std(log_returns over 14 days)
- Purpose: Medium-term volatility feature
- Mean: 0.057173

### 4-6. Moving Averages (7, 14, 30-day)
- Formula: mean(prices over window)
- Purpose: Capture price trends
- Correlation with target: MA7 = -0.043

### 7. Bollinger Band Width
- Formula: (Upper - Lower) / Middle
- Purpose: Measure volatility
- Correlation with target: 0.598

### 8. Average True Range (ATR)
- Formula: 14-day average of true range
- Purpose: Volatility indicator
- Correlation with target: -0.020

### 9. Liquidity Ratio
- Formula: Volume / Market Cap
- Purpose: Trading activity measure
- Correlation with target: -0.060

### 10. Momentum (7-day)
- Formula: Price_t / Price_t-7
- Purpose: Price momentum indicator
- Correlation with target: 0.205

## Data Quality
- Initial rows: 69,841
- Final rows: 68,245
- Rows removed (NaN): 1,596
- Final columns: 23

## Next Steps
1. Model training with Random Forest
2. Hyperparameter optimization
3. Model evaluation
4. Deployment
