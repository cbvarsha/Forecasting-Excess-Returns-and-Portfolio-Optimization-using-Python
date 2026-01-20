#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import matplotlib.dates as mdates

# Function to compute annualized statistics
def calculate_statistics(returns):
    annualized_mean = returns.mean() * 12
    annualized_volatility = returns.std() * np.sqrt(12)
    sharpe_ratio = annualized_mean / annualized_volatility
    skewness = skew(returns)
    kurt = kurtosis(returns)
    return annualized_mean, annualized_volatility, sharpe_ratio, skewness, kurt

# Load the dataset
file_path = 'Datas1.xlsx'  # Replace with your file path

# Check column names in the file to identify the date column
data = pd.read_excel(file_path)  # Load without parsing first to inspect columns
print(data.columns)  # Print column names to find the actual date column name

# Update the date column name as per the file's content
data = pd.read_excel(file_path, parse_dates=['Dates'])  # Replace 'YourDateColumnName' with the correct column name

# Set Date as index
data.set_index('Dates', inplace=True)  # Replace with correct column name

# Clean column names
data.columns = data.columns.str.strip()

# Calculate monthly returns for stocks and bonds
data['Stock Monthly Returns'] = data['stock index price'].pct_change()
data['Bond Monthly Returns'] = data['Bond index price'].pct_change()

# Calculate excess returns
data['Stock Excess Returns'] = data['Stock Monthly Returns'] - data['Risk free rate of return']
data['Bond Excess Returns'] = data['Bond Monthly Returns'] - data['Risk free rate of return']

# Remove rows with NaN values (from pct_change or missing risk-free rates)
cleaned_data = data.dropna(subset=['Stock Monthly Returns', 'Bond Monthly Returns', 
                                   'Stock Excess Returns', 'Bond Excess Returns'])

# Compute statistics for stock and bond excess returns
stock_stats = calculate_statistics(cleaned_data['Stock Excess Returns'])
bond_stats = calculate_statistics(cleaned_data['Bond Excess Returns'])

# Create a summary table for statistics
summary_table = pd.DataFrame({
    'Statistic': ['Annualized Mean', 'Annualized Volatility', 'Annualized Sharpe Ratio', 'Skewness', 'Kurtosis'],
    'Stock': stock_stats,
    'Bond': bond_stats
})

# Display the summary table
print("Summary Statistics for Excess Returns")
print(summary_table)

# Save computed data and statistics to an Excel file
with pd.ExcelWriter('Full_Results.xlsx') as writer:
    # Save computed returns and excess returns
    cleaned_data.to_excel(writer, sheet_name='Computed Returns', index=True)
    # Save summary statistics
    summary_table.to_excel(writer, sheet_name='Summary Statistics', index=False)

# Plot the excess returns
plt.figure(figsize=(10, 6))
plt.plot(cleaned_data['Stock Excess Returns'], label='Stock Excess Returns')
plt.plot(cleaned_data['Bond Excess Returns'], label='Bond Excess Returns')
plt.title("Excess Returns of S&P 500 and Bonds")
plt.ylabel("Excess Return")
plt.xlabel("Date")
plt.legend()
plt.grid()

# Format the x-axis dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)
plt.show()


# In[63]:


#Q2 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to compute rolling mean forecast
def rolling_mean_forecast(excess_returns, window_size):
    """
    Compute rolling mean forecast for a given series of excess returns.
    
    Parameters:
    - excess_returns (pd.Series): Time series of excess returns.
    - window_size (int): Size of the rolling window.

    Returns:
    - pd.Series: Rolling mean forecast.
    """
    return excess_returns.rolling(window=window_size).mean().shift(1)

# Load dataset
file_path = 'Full_Results.xlsx'
data = pd.read_excel(file_path, sheet_name='Computed Returns')

# Ensure proper date formatting and set as index
data['Date'] = pd.to_datetime(data['Dates'])
data.set_index('Date', inplace=True)

# Divide data into in-sample and out-of-sample periods
in_sample = data.loc['1980-01-01':'2000-12-31']
out_of_sample = data.loc['2001-01-01':'2023-12-31']

# Rolling window size (e.g., 60 months = 5 years)
window_size = 60

# Generate rolling mean forecasts for stocks and bonds
stock_forecast = rolling_mean_forecast(out_of_sample['Stock Excess Returns'], window_size)
bond_forecast = rolling_mean_forecast(out_of_sample['Bond Excess Returns'], window_size)

# Combine forecasts with out-of-sample data
forecasts = out_of_sample[['Stock Excess Returns', 'Bond Excess Returns']].copy()
forecasts['Stock Forecast'] = stock_forecast
forecasts['Bond Forecast'] = bond_forecast

# Save results to Excel
with pd.ExcelWriter('Rolling_Forecasts.xlsx', engine='openpyxl') as writer:
    in_sample.to_excel(writer, sheet_name='In-Sample Data', index=True)
    out_of_sample.to_excel(writer, sheet_name='Out-of-Sample Data', index=True)
    forecasts.to_excel(writer, sheet_name='Forecasts', index=True)

# Plot rolling forecasts
plt.figure(figsize=(10, 6))
plt.plot(forecasts.index, forecasts['Stock Forecast'], label='Stock Forecast', color='blue')
plt.plot(forecasts.index, forecasts['Bond Forecast'], label='Bond Forecast', color='orange')
plt.title("Rolling Mean Forecasts of Excess Returns")
plt.xlabel("Date")
plt.ylabel("Excess Return Forecast")
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.show()

    # Plot actual and benchmark forecasts for stocks
plt.figure(figsize=(12, 6))
plt.plot(forecasts.index, forecasts['Stock Excess Returns'], label='Actual Stock Excess Returns', color='blue')
plt.plot(forecasts.index, forecasts['Stock Forecast'], label='Stock Benchmark Forecast', color='green')
plt.title('Stock Excess Returns: Actual vs. Benchmark Forecast')
plt.xlabel('Date')
plt.ylabel('Excess Return')
plt.legend()
plt.grid()
plt.show()

# Plot actual and benchmark forecasts for bonds
plt.figure(figsize=(12, 6))
plt.plot(forecasts.index, forecasts['Bond Excess Returns'], label='Actual Bond Excess Returns', color='orange')
plt.plot(forecasts.index, forecasts['Bond Forecast'], label='Bond Benchmark Forecast', color='purple')
plt.title('Bond Excess Returns: Actual vs. Benchmark Forecast')
plt.xlabel('Date')
plt.ylabel('Excess Return')
plt.legend()
plt.grid()
plt.show()

# Display results
print("In-Sample Data:")
print(in_sample.head())
print("\nOut-of-Sample Data:")
print(out_of_sample.head())
print("\nForecasts:")
print(forecasts.head())


# In[54]:


#Q3 

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load the data
file_path = 'PredictorData2023.xlsx'  # Replace with the correct path
data = pd.read_excel(file_path, sheet_name='Monthly')

# Clean column names
data.columns = data.columns.str.strip()

# Replace with your actual column names for stock, bond returns, and risk-free rates
stock_returns_col = 'ret'  # Stock total returns
bond_returns_col = 'corpr'  # Corporate bond returns
risk_free_rate_col = 'Rfree'  # Risk-free rate

# Specify predictors (replace with your selected predictor columns)
predictors = ['d12', 'e12', 'lty', 'csp', 'vrp']

# Drop rows with NaN values in required columns
required_columns = predictors + [stock_returns_col, bond_returns_col, risk_free_rate_col]
data = data.dropna(subset=required_columns)

# Calculate excess returns
data['Stock_Excess_Returns'] = data[stock_returns_col] - data[risk_free_rate_col]
data['Bond_Excess_Returns'] = data[bond_returns_col] - data[risk_free_rate_col]

# Ensure predictors exist in the dataset
for predictor in predictors:
    if predictor not in data.columns:
        raise ValueError(f"Predictor '{predictor}' is not in the dataset.")

# Rolling window size
window_size = 60

# Function to generate OLS forecasts
def ols_forecast(predictors, target, window):
    forecasts = []
    for i in range(len(target) - window):
        train_X = predictors.iloc[i:i + window].values
        train_y = target.iloc[i:i + window].values
        test_X = predictors.iloc[i + window:i + window + 1].values

        model = LinearRegression()
        model.fit(train_X, train_y)
        forecasts.append(model.predict(test_X)[0])
    return pd.Series(forecasts, index=target.index[window:])

# Generate OLS forecasts for stock and bond excess returns
ols_forecasts_stock = {}
ols_forecasts_bond = {}

for predictor in predictors:
    ols_forecasts_stock[predictor] = ols_forecast(data[[predictor]], data['Stock_Excess_Returns'], window_size)
    ols_forecasts_bond[predictor] = ols_forecast(data[[predictor]], data['Bond_Excess_Returns'], window_size)

# Combination forecast: Average of OLS forecasts
ols_combination_stock = pd.DataFrame(ols_forecasts_stock).mean(axis=1)
ols_combination_bond = pd.DataFrame(ols_forecasts_bond).mean(axis=1)

# Benchmark forecasts (rolling mean forecast)
benchmark_stock = data['Stock_Excess_Returns'].rolling(window=window_size).mean().shift(1).dropna()
benchmark_bond = data['Bond_Excess_Returns'].rolling(window=window_size).mean().shift(1).dropna()

# Calculate MSFE (Mean Squared Forecast Error)
def calculate_msfe(actual, forecast):
    errors = actual - forecast  # Forecast errors (e_t)
    squared_errors = errors ** 2  # Square of errors (e_t^2)
    return np.mean(squared_errors)  # Mean of squared errors

# MSFE for stock and bond forecasts
msfe_benchmark_stock = calculate_msfe(data['Stock_Excess_Returns'][window_size:], benchmark_stock)
msfe_benchmark_bond = calculate_msfe(data['Bond_Excess_Returns'][window_size:], benchmark_bond)

msfe_results_stock = {predictor: calculate_msfe(data['Stock_Excess_Returns'][window_size:], forecast) 
                      for predictor, forecast in ols_forecasts_stock.items()}
msfe_results_bond = {predictor: calculate_msfe(data['Bond_Excess_Returns'][window_size:], forecast) 
                     for predictor, forecast in ols_forecasts_bond.items()}

msfe_combination_stock = calculate_msfe(data['Stock_Excess_Returns'][window_size:], ols_combination_stock)
msfe_combination_bond = calculate_msfe(data['Bond_Excess_Returns'][window_size:], ols_combination_bond)

# Diebold-Mariano test
def diebold_mariano_test(actual, forecast1, forecast2):
    e1 = actual - forecast1
    e2 = actual - forecast2
    d = (e1 ** 2) - (e2 ** 2)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)  # Sample variance of d
    dm_stat = mean_d / np.sqrt(var_d / len(d))  # DM statistic
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))  # Two-tailed p-value
    return dm_stat, p_value

# DM Test for Combination Forecast vs Benchmark
actual_stock = data['Stock_Excess_Returns'][window_size:]
actual_bond = data['Bond_Excess_Returns'][window_size:]

dm_stat_stock, p_value_stock = diebold_mariano_test(actual_stock, ols_combination_stock, benchmark_stock)
dm_stat_bond, p_value_bond = diebold_mariano_test(actual_bond, ols_combination_bond, benchmark_bond)

# Visualization of Stock Excess Returns Forecasts
plt.figure(figsize=(14, 6))
plt.plot(data.index[window_size:], data['Stock_Excess_Returns'][window_size:], label='Actual Stock Excess Returns', color='blue')
plt.plot(data.index[window_size:], ols_combination_stock, label='Model Forecast', color='orange')
plt.plot(data.index[window_size:], benchmark_stock, label='Benchmark Forecast', color='green')
plt.legend()
plt.title('Stock Excess Returns Forecasts')
plt.xlabel('Date')
plt.ylabel('Excess Returns')
plt.show()

# Visualization of Bond Excess Returns Forecasts
plt.figure(figsize=(14, 6))
plt.plot(data.index[window_size:], data['Bond_Excess_Returns'][window_size:], label='Actual Bond Excess Returns', color='blue')
plt.plot(data.index[window_size:], ols_combination_bond, label='Model Forecast', color='orange')
plt.plot(data.index[window_size:], benchmark_bond, label='Benchmark Forecast', color='green')
plt.legend()
plt.title('Bond Excess Returns Forecasts')
plt.xlabel('Date')
plt.ylabel('Excess Returns')
plt.show()

# Print Results
print("Stock MSFE Results:")
print(f"Benchmark: {msfe_benchmark_stock}")
for predictor, msfe in msfe_results_stock.items():
    print(f"{predictor}: {msfe}")
print(f"Combination Forecast: {msfe_combination_stock}")
print(f"Diebold-Mariano Test Statistic (Stock): {dm_stat_stock}, P-value: {p_value_stock}")

print("\nBond MSFE Results:")
print(f"Benchmark: {msfe_benchmark_bond}")
for predictor, msfe in msfe_results_bond.items():
    print(f"{predictor}: {msfe}")
print(f"Combination Forecast: {msfe_combination_bond}")
print(f"Diebold-Mariano Test Statistic (Bond): {dm_stat_bond}, P-value: {p_value_bond}")

# Save results to an Excel file
output_file = 'Forecast_Results.xlsx'
with pd.ExcelWriter(output_file) as writer:
    pd.DataFrame(ols_forecasts_stock).to_excel(writer, sheet_name='Stock_OLS_Forecasts')
    pd.DataFrame(ols_forecasts_bond).to_excel(writer, sheet_name='Bond_OLS_Forecasts')
    pd.DataFrame({'Combination Forecast (Stock)': ols_combination_stock}).to_excel(writer, sheet_name='Stock_Combination_Forecast')
    pd.DataFrame({'Combination Forecast (Bond)': ols_combination_bond}).to_excel(writer, sheet_name='Bond_Combination_Forecast')
    benchmark_stock.to_excel(writer, sheet_name='Stock_Benchmark_Forecast')
    benchmark_bond.to_excel(writer, sheet_name='Bond_Benchmark_Forecast')


# In[65]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = 'PredictorData2023.xlsx'  # Replace with the correct path to your file
data = pd.read_excel(file_path, sheet_name='Monthly')

# Clean column names
data.columns = data.columns.str.strip()

# Define columns for stock, bond returns, and risk-free rate
stock_returns_col = 'ret'  # Stock total returns
bond_returns_col = 'corpr'  # Corporate bond returns
risk_free_rate_col = 'Rfree'  # Risk-free rate

# Ensure required columns exist
required_columns = [stock_returns_col, bond_returns_col, risk_free_rate_col]
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Column '{col}' is missing in the dataset.")

# Drop NaN values in relevant columns
data = data.dropna(subset=required_columns)

# Calculate stock and bond excess returns
data['Stock_Excess_Returns'] = data[stock_returns_col] - data[risk_free_rate_col]
data['Bond_Excess_Returns'] = data[bond_returns_col] - data[risk_free_rate_col]

# Rolling window size
window_size = 60

# Function to generate rolling variance-covariance matrix
def rolling_variance_covariance(data, stock_col, bond_col, window):
    variance_covariance_matrices = []
    dates = []
    for i in range(len(data) - window + 1):
        rolling_data = data.iloc[i:i + window]
        covariance_matrix = rolling_data[[stock_col, bond_col]].cov()
        variance_covariance_matrices.append(covariance_matrix.values)
        dates.append(data.index[i + window - 1])
    return variance_covariance_matrices, dates

# Generate rolling variance-covariance matrices
variance_cov_matrices, matrix_dates = rolling_variance_covariance(
    data,
    'Stock_Excess_Returns',
    'Bond_Excess_Returns',
    window_size
)

# Save results to a DataFrame for visualization or export
results = []
for i, date in enumerate(matrix_dates):
    matrix = variance_cov_matrices[i]
    results.append({
        'Date': date,
        'Var_Stock': matrix[0, 0],
        'Var_Bond': matrix[1, 1],
        'Cov_Stock_Bond': matrix[0, 1]
    })

variance_cov_df = pd.DataFrame(results)

# Save results to an Excel file
output_file = 'Variance_Covariance_Results.xlsx'
variance_cov_df.to_excel(output_file, index=False)

# Plot rolling variances and covariance
plt.figure(figsize=(12, 6))
plt.plot(variance_cov_df['Date'], variance_cov_df['Var_Stock'], label='Rolling Stock Variance', color='blue')
plt.plot(variance_cov_df['Date'], variance_cov_df['Var_Bond'], label='Rolling Bond Variance', color='green')
plt.plot(variance_cov_df['Date'], variance_cov_df['Cov_Stock_Bond'], label='Rolling Stock-Bond Covariance', color='orange')
plt.title('Rolling Variance-Covariance Matrix Dynamics')
plt.xlabel('Date')
plt.ylabel('Variance / Covariance')
plt.legend()
plt.grid()
plt.show()

# Display results in Python
print("Rolling Variance-Covariance Matrices")
print(variance_cov_df.head())


# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = 'PredictorData2023.xlsx'  # Replace with the correct path
data = pd.read_excel(file_path, sheet_name='Monthly')

# Clean column names
data.columns = data.columns.str.strip()

# Define columns for stock, bond returns, and risk-free rate
stock_returns_col = 'ret'  # Stock total returns
bond_returns_col = 'corpr'  # Corporate bond returns
risk_free_rate_col = 'Rfree'  # Risk-free rate

# Ensure required columns exist
required_columns = [stock_returns_col, bond_returns_col, risk_free_rate_col]
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Column '{col}' is missing in the dataset.")

# Drop NaN values in relevant columns
data = data.dropna(subset=required_columns)

# Calculate stock and bond excess returns
data['Stock_Excess_Returns'] = data[stock_returns_col] - data[risk_free_rate_col]
data['Bond_Excess_Returns'] = data[bond_returns_col] - data[risk_free_rate_col]

# Rolling window size
window_size = 60
lambda_risk_aversion = 3

# Function to generate rolling variance-covariance matrix
def rolling_variance_covariance(data, stock_col, bond_col, window):
    variance_covariance_matrices = []
    dates = []
    for i in range(len(data) - window):
        rolling_data = data.iloc[i:i + window]
        covariance_matrix = rolling_data[[stock_col, bond_col]].cov()
        variance_covariance_matrices.append(covariance_matrix.values)
        dates.append(data.index[i + window])
    return variance_covariance_matrices, dates

# Generate rolling variance-covariance matrices
variance_cov_matrices, matrix_dates = rolling_variance_covariance(
    data,
    'Stock_Excess_Returns',
    'Bond_Excess_Returns',
    window_size
)

# Placeholder for benchmark forecasts (use historical mean for simplicity)
data['Benchmark_Stock'] = data['Stock_Excess_Returns'].rolling(window=window_size).mean().shift(1)
data['Benchmark_Bond'] = data['Bond_Excess_Returns'].rolling(window=window_size).mean().shift(1)
benchmark_forecasts = data[['Benchmark_Stock', 'Benchmark_Bond']].dropna()

# Placeholder for model-based forecasts (add actual model results here)
data['Model_Stock'] = data['Benchmark_Stock'] * 1.05  # Example multiplier for model-based forecasts
data['Model_Bond'] = data['Benchmark_Bond'] * 0.95
model_forecasts = data[['Model_Stock', 'Model_Bond']].dropna()

# Compute optimal portfolio weights and portfolio returns (Benchmark)
benchmark_weights = []
benchmark_returns = []

for i in range(len(benchmark_forecasts)):
    mu_t = benchmark_forecasts.iloc[i].values.reshape(-1, 1)  # Forecast returns
    sigma_t = variance_cov_matrices[i]  # Variance-covariance matrix
    sigma_inv = np.linalg.inv(sigma_t)  # Inverse of covariance matrix
    w_t = (1 / lambda_risk_aversion) * np.dot(sigma_inv, mu_t)  # Optimal weights

    # Actual realized returns for t+1
    realized_returns = data.iloc[window_size + i][['Stock_Excess_Returns', 'Bond_Excess_Returns']].values
    portfolio_return = np.dot(w_t.T, realized_returns)

    benchmark_weights.append(w_t.flatten())
    benchmark_returns.append(portfolio_return[0])

# Compute optimal portfolio weights and portfolio returns (Model-Based)
model_weights = []
model_returns = []

for i in range(len(model_forecasts)):
    mu_t = model_forecasts.iloc[i].values.reshape(-1, 1)  # Forecast returns
    sigma_t = variance_cov_matrices[i]  # Variance-covariance matrix
    sigma_inv = np.linalg.inv(sigma_t)  # Inverse of covariance matrix
    w_t = (1 / lambda_risk_aversion) * np.dot(sigma_inv, mu_t)  # Optimal weights

    # Actual realized returns for t+1
    realized_returns = data.iloc[window_size + i][['Stock_Excess_Returns', 'Bond_Excess_Returns']].values
    portfolio_return = np.dot(w_t.T, realized_returns)

    model_weights.append(w_t.flatten())
    model_returns.append(portfolio_return[0])

# Convert weights and returns to DataFrames
benchmark_weights_df = pd.DataFrame(benchmark_weights, columns=['Weight_Stock', 'Weight_Bond'], index=benchmark_forecasts.index)
model_weights_df = pd.DataFrame(model_weights, columns=['Weight_Stock', 'Weight_Bond'], index=model_forecasts.index)
benchmark_returns_df = pd.DataFrame(benchmark_returns, columns=['Portfolio_Return'], index=benchmark_forecasts.index)
model_returns_df = pd.DataFrame(model_returns, columns=['Portfolio_Return'], index=model_forecasts.index)

# Compute annualized portfolio statistics (Benchmark)
benchmark_annualized_mean = benchmark_returns_df.mean() * 12
benchmark_annualized_volatility = benchmark_returns_df.std() * np.sqrt(12)
benchmark_sharpe_ratio = benchmark_annualized_mean / benchmark_annualized_volatility

# Compute annualized portfolio statistics (Model-Based)
model_annualized_mean = model_returns_df.mean() * 12
model_annualized_volatility = model_returns_df.std() * np.sqrt(12)
model_sharpe_ratio = model_annualized_mean / model_annualized_volatility

# Display portfolio statistics
print("Benchmark Portfolio Annualized Statistics")
print(f"Mean: {benchmark_annualized_mean.values[0]:.4f}")
print(f"Volatility: {benchmark_annualized_volatility.values[0]:.4f}")
print(f"Sharpe Ratio: {benchmark_sharpe_ratio.values[0]:.4f}")

print("\nModel-Based Portfolio Annualized Statistics")
print(f"Mean: {model_annualized_mean.values[0]:.4f}")
print(f"Volatility: {model_annualized_volatility.values[0]:.4f}")
print(f"Sharpe Ratio: {model_sharpe_ratio.values[0]:.4f}")

# Plot portfolio weights and cumulative returns (Benchmark and Model-Based)
benchmark_cumulative_returns = (1 + benchmark_returns_df).cumprod()
model_cumulative_returns = (1 + model_returns_df).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(benchmark_cumulative_returns, label='Benchmark Portfolio Return', color='blue')
plt.plot(model_cumulative_returns, label='Model-Based Portfolio Return', color='orange')
plt.title('Cumulative Portfolio Returns: Benchmark vs Model-Based')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(benchmark_weights_df['Weight_Stock'], label='Benchmark Stock Weight', color='blue', linestyle='--')
plt.plot(benchmark_weights_df['Weight_Bond'], label='Benchmark Bond Weight', color='green', linestyle='--')
plt.plot(model_weights_df['Weight_Stock'], label='Model-Based Stock Weight', color='blue')
plt.plot(model_weights_df['Weight_Bond'], label='Model-Based Bond Weight', color='green')
plt.title('Portfolio Weights: Benchmark vs Model-Based')
plt.xlabel('Date')
plt.ylabel('Portfolio Weights')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




