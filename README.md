# Forecasting Excess Returns and Portfolio Optimization using Python

##  Overview
This project applies predictive modeling and portfolio optimization techniques to forecast excess returns of equities (S&P 500) and bonds (Bloomberg Barclays Aggregate Bond Index) using monthly data from **1979–2023**.

The study evaluates whether econometric forecasting models improve **out-of-sample forecast accuracy** and **risk-adjusted portfolio performance** compared to simple benchmark forecasts.

---

##  Objectives
- Analyze the statistical properties of stock and bond excess returns
- Generate benchmark forecasts using rolling historical means
- Build predictive models using economic indicators (OLS regression)
- Evaluate forecast performance using MSFE and Diebold–Mariano tests
- Construct dynamically optimized portfolios using mean–variance optimization

---

##  Data Description
- **Equities:** S&P 500 Index (monthly returns)
- **Bonds:** Bloomberg Barclays U.S. Aggregate Bond Index
- **Risk-Free Rate:** Short-term Treasury rate
- **Predictors:**  
  - Dividend–price ratio  
  - Earnings–price ratio  
  - Long-term yield  
  - Credit spread  
  - Variance risk premium  

**Time Period:** December 1979 – December 2023

> ⚠️ Raw data files are not included due to licensing restrictions.

---

##  Methodology
1. **Data preprocessing**
   - Calculation of monthly returns and excess returns
2. **Descriptive statistics**
   - Annualized mean, volatility, Sharpe ratio, skewness, kurtosis
3. **Benchmark forecasting**
   - Rolling 60-month historical mean forecasts
4. **Predictive modeling**
   - OLS regression with individual predictors
   - Combination (average) forecasts
5. **Forecast evaluation**
   - Mean Squared Forecast Error (MSFE)
   - Diebold–Mariano statistical test
6. **Portfolio optimization**
   - Rolling variance–covariance estimation
   - Mean–variance optimization with risk aversion parameter

---

##  Project Structure
```

src/        → Modular Python scripts
data/       → Raw and processed datasets
Report/    → Figures and result tables

````

---

##  Key Findings
- Combination forecasts achieved **lower MSFE** than benchmark forecasts
- Statistical improvements were economically meaningful but not always statistically significant
- Model-based portfolios showed:
  - Higher Sharpe ratios
  - Lower volatility
  - Improved risk-adjusted performance

---

---

## Author Contribution

* Data preprocessing and feature engineering
* Forecast model development and evaluation
* Portfolio optimization implementation
* Results interpretation and visualization

---

## Future Improvements

* Incorporate machine learning models (Random Forest, LSTM)
* Include transaction costs and rebalancing constraints
* Expand analysis to additional asset classes

---

## Disclaimer

This project is for **academic and portfolio demonstration purposes only** and does not constitute financial advice.
