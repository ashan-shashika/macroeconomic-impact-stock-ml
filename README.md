# Macroeconomic Determinants of S&P 500 Returns
### A Machine Learning Study with Sector-Level SHAP Analysis

This project looks into whether machine learning models can use macroeconomic variables to predict monthly S&P 500 returns. Used seven macro variables—inflation, interest rates, GDP growth, unemployment, the US dollar index, market volatility (VIX), and credit spreads to train and compare five model families: Linear Regression, Ridge Regression, Random Forest, XGBoost, and Neural Networks.

Then, the best models are used on five S&P 500 sector ETFs: Energy, Technology, Financials, Healthcare, and Industrials. To find out which macroeconomic factors affect each sector in a different way, SHAP values and built in feature importance are compared across sectors.

**MSc Data Science — University of Hertfordshire**

---

## Project Highlights

- 11 models trained and evaluated on S&P 500 monthly returns (1990–2025)
- Hyperparameter tuning with Optuna Bayesian search and cross validation
- Sector level SHAP analysis across Energy, Technology, Financials, Healthcare, Industrials
- VIX identified as the dominant macro predictor across all sectors

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/ashan-shashika/macroeconomic-impact-stock-ml.git
cd macroeconomic-impact-stock-ml
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install required libraries**
```bash
pip install -r requirements.txt
```

**4. Add your FRED API key**

Get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html

Create a file called `.env` in the root folder and add:
```
FRED_API_KEY=your_api_key_here
```

**5. Run notebooks in order**
```bash
jupyter notebook
```

Open Jupyter and run notebooks sequentially from 01 to 10.




