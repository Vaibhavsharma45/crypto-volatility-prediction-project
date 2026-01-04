# Project Setup Guide
## Cryptocurrency Volatility Prediction System

Complete step-by-step instructions to set up and run the project.

---

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ Python 3.8 or higher installed
- ✅ pip (Python package manager)
- ✅ Git (for cloning repository)
- ✅ 4GB RAM minimum
- ✅ 500MB free disk space

---

## 🚀 Quick Start (5 Minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/crypto-volatility-prediction.git
cd crypto-volatility-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python src/preprocessing.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate_model.py

# Launch app
streamlit run app.py
```

---

## 📦 Detailed Setup Instructions

### Step 1: System Requirements Check

#### Check Python Version
```bash
python --version
# Should show: Python 3.8.x or higher
```

#### Check pip
```bash
pip --version
# Should show: pip 20.x or higher
```

### Step 2: Create Project Directory

```bash
# Option A: Clone from GitHub (if published)
git clone https://github.com/yourusername/crypto-volatility-prediction.git
cd crypto-volatility-prediction

# Option B: Create manually
mkdir crypto-volatility-prediction
cd crypto-volatility-prediction
```

### Step 3: Create Folder Structure

```bash
# Create all required directories
mkdir -p data/raw data/processed model reports notebooks src

# Verify structure
tree -L 2  # On Windows: use dir /s
```

Expected structure:
```
crypto-volatility-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── model/
├── reports/
├── notebooks/
└── src/
```

### Step 4: Set Up Virtual Environment

#### Why Virtual Environment?
- Isolates project dependencies
- Prevents package conflicts
- Makes deployment easier

#### Create Virtual Environment

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Verify activation:**
```bash
which python  # Should show venv/bin/python
```

### Step 5: Install Dependencies

#### Create requirements.txt
Copy the requirements.txt file provided or create with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit jupyter
pip freeze > requirements.txt
```

#### Install from requirements.txt
```bash
pip install -r requirements.txt
```

#### Verify Installation
```bash
pip list
# Should show all packages installed
```

### Step 6: Place Your Data

```bash
# Copy your dataset to the raw data folder
cp /path/to/your/dataset.csv data/raw/dataset.csv

# Verify
ls -lh data/raw/dataset.csv
```

**Important:** Dataset must have these columns:
- date
- crypto_name
- open
- high
- low
- close
- volume
- marketCap

### Step 7: Copy Source Files

Place all Python files in their respective directories:

```
src/
├── preprocessing.py
├── feature_engineering.py
├── train_model.py
└── evaluate_model.py

# Root directory:
app.py
requirements.txt
README.md
```

---

## ⚙️ Running the Pipeline

### Complete Pipeline Execution

```bash
# Step 1: Data Preprocessing (2-5 seconds)
python src/preprocessing.py
# Output: data/processed/cleaned_data.csv

# Step 2: Feature Engineering (5-10 seconds)
python src/feature_engineering.py
# Output: data/processed/features.csv

# Step 3: Model Training (5-15 minutes)
python src/train_model.py
# Output: model/volatility_model.pkl
#         model/test_data.pkl

# Step 4: Model Evaluation (10-20 seconds)
python src/evaluate_model.py
# Output: reports/model_evaluation.png
#         reports/feature_importance.png
#         reports/evaluation_metrics.txt
```

### Check Outputs After Each Step

```bash
# After preprocessing
ls -lh data/processed/cleaned_data.csv

# After feature engineering
ls -lh data/processed/features.csv

# After training
ls -lh model/*.pkl

# After evaluation
ls -lh reports/*.png
```

---

## 🌐 Running the Streamlit App

### Launch Application

```bash
streamlit run app.py
```

### What Happens Next

1. Terminal shows: `You can now view your Streamlit app in your browser`
2. Browser opens automatically at `http://localhost:8501`
3. If not, manually open the URL shown in terminal

### Using the App

1. **Upload Data:** Click "Browse files" and upload your CSV
2. **Select Crypto:** Choose from dropdown
3. **View Overview:** See market statistics
4. **Predict:** Click "Predict Volatility" button
5. **Analyze:** View charts and statistics
6. **Download:** Export predictions as CSV

### Stop the App

Press `Ctrl+C` in the terminal

---

## 🧪 Running Jupyter Notebooks (Optional)

### Start Jupyter

```bash
jupyter notebook
```

### Create and Run Notebooks

1. Navigate to `notebooks/` directory
2. Create new notebook: `01_eda.ipynb`
3. Add cells for exploratory analysis
4. Run cells to visualize data

### Example Notebook Cell

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('../data/processed/cleaned_data.csv')

# Quick visualization
df.groupby('crypto_name')['close'].plot(legend=True)
plt.title('Price Trends by Cryptocurrency')
plt.show()
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Module Not Found
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```bash
pip install pandas
# Or reinstall all
pip install -r requirements.txt
```

#### Issue 2: File Not Found
```
FileNotFoundError: data/raw/dataset.csv not found
```

**Solution:**
```bash
# Check file location
ls data/raw/

# Copy file to correct location
cp /path/to/dataset.csv data/raw/
```

#### Issue 3: Permission Denied
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Make directory writable
chmod -R 755 data/ model/ reports/

# Or run with appropriate permissions
sudo python src/preprocessing.py  # Use carefully
```

#### Issue 4: Memory Error
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# In your code, process data in chunks
df = pd.read_csv('file.csv', chunksize=10000)
```

#### Issue 5: Streamlit Port in Use
```
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process
lsof -ti:8501 | xargs kill -9  # Linux/Mac
# Windows: Use Task Manager to end Python process
```

#### Issue 6: Model File Not Found
```
FileNotFoundError: model/volatility_model.pkl not found
```

**Solution:**
```bash
# Train model first
python src/train_model.py
```

---

## 🔧 Configuration Options

### Customize Training Parameters

Edit `src/train_model.py`:

```python
# Modify train-test split ratio
train_size = 0.80  # Change to 0.70 for 70/30 split

# Modify hyperparameter grid
param_grid = {
    'n_estimators': [50, 100],  # Fewer values for faster training
    'max_depth': [20, None],     # Reduce combinations
}
```

### Customize App Settings

Edit `app.py`:

```python
# Change page configuration
st.set_page_config(
    page_title="My Crypto Predictor",  # Custom title
    page_icon="💰",                    # Custom icon
)
```

---

## 📊 Verifying Installation

### Run Verification Script

Create `verify_setup.py`:

```python
#!/usr/bin/env python3

import sys

print("Checking Python version...")
print(f"Python {sys.version}")

print("\nChecking installed packages...")
packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'plotly', 'streamlit']

for package in packages:
    try:
        __import__(package)
        print(f"✅ {package} installed")
    except ImportError:
        print(f"❌ {package} NOT installed")

print("\nChecking directory structure...")
import os

dirs = ['data/raw', 'data/processed', 'model', 'reports', 'src']
for d in dirs:
    if os.path.exists(d):
        print(f"✅ {d}/ exists")
    else:
        print(f"❌ {d}/ NOT found")

print("\n✅ Setup verification complete!")
```

Run it:
```bash
python verify_setup.py
```

---

## 🔄 Updating the Project

### Pull Latest Changes (if using Git)

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Reinstall Dependencies

```bash
pip install -r requirements.txt --upgrade --force-reinstall
```

---

## 🗑️ Cleanup

### Remove Virtual Environment

```bash
deactivate  # Exit virtual environment
rm -rf venv/  # Delete virtual environment
```

### Remove Generated Files

```bash
# Remove processed data
rm -rf data/processed/*

# Remove models
rm -rf model/*

# Remove reports
rm -rf reports/*
```

### Complete Cleanup

```bash
# Be careful! This removes everything
rm -rf data/ model/ reports/ venv/
```

---

## 📖 Additional Resources

### Learning Resources
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [pandas Documentation](https://pandas.pydata.org/)

### Helpful Commands

```bash
# List installed packages
pip list

# Check package version
pip show pandas

# Update single package
pip install --upgrade pandas

# Export environment
pip freeze > requirements.txt

# Create .gitignore
echo "venv/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
```

---

## ✅ Success Checklist

Before considering setup complete, verify:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] Directory structure created
- [ ] Dataset placed in `data/raw/`
- [ ] All source files in correct locations
- [ ] Preprocessing runs successfully
- [ ] Feature engineering runs successfully
- [ ] Model training completes
- [ ] Model evaluation generates reports
- [ ] Streamlit app launches
- [ ] Can upload data and get predictions

---

## 🆘 Getting Help

If you encounter issues:

1. **Check Error Messages:** Read carefully, they often explain the problem
2. **Check File Locations:** Ensure all files are in correct directories
3. **Check Python Version:** Must be 3.8+
4. **Check Virtual Environment:** Should be activated
5. **Reinstall Dependencies:** Sometimes fixes corruption

---

## 🎉 Next Steps

Once setup is complete:

1. ✅ Run the complete pipeline
2. ✅ Explore the Streamlit app
3. ✅ Read the documentation
4. ✅ Customize for your needs
5. ✅ Deploy to production (optional)

---

**Setup Guide Version:** 1.0  
**Last Updated:** January 2025  
**Maintained by:** PW Skills Student

**Happy Coding! 🚀**