import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.optimize as sc
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns
from scipy.stats import kurtosis, skew
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def download_data(etfs, start_date, end_date):
    data = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
    return data

def preprocess_data(data):
    data_interpolated = data.interpolate(method='linear')
    data_returns = data_interpolated.pct_change().dropna()
    return data_returns

def calculate_statistics(data_returns):
    mean_returns = data_returns.mean()
    cov_matrix = data_returns.cov()
    return mean_returns, cov_matrix

def plot_historical_prices(data, etfs):
    plt.figure(figsize=(14, 5))
    for etf in etfs:
        plt.plot(data.index, data[etf], label=etf)
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.title('Historical Prices of ETFs')
    plt.legend()
    plt.grid(False)
    plt.show()

def decompose_time_series(data, etf, period=252):
    result = seasonal_decompose(data[etf], model='multiplicative', period=period)
    result.plot()
    plt.show()

def perform_adf_test(data, etfs):
    for etf in etfs:
        print(f"\nResults for {etf}:")
        adf_result = adfuller(data[etf])
        print('ADF Statistic:', adf_result[0])
        print('p-value:', adf_result[1])
        
        if adf_result[1] > 0.05:
            data_diff = data[etf].diff().dropna()
            adf_result_diff = adfuller(data_diff)
            print('ADF Statistic after differencing:', adf_result_diff[0])
            print('p-value after differencing:', adf_result_diff[1])
            plot_acf_pacf(data_diff, etf, differenced=True)
        else:
            plot_acf_pacf(data[etf], etf)

def plot_acf_pacf(data_series, etf, differenced=False):
    suffix = 'Differenced ' if differenced else ''
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(data_series, ax=ax[0], title=f'ACF for {suffix}{etf}')
    plot_pacf(data_series, ax=ax[1], title=f'PACF for {suffix}{etf}')
    plt.tight_layout()
    plt.show()

def plot_box_plots(data_returns):
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=data_returns)
    plt.title('Box Plots of Daily Return Distributions for ETFs')
    plt.xlabel('ETF')
    plt.ylabel('Daily Return')
    plt.grid(True)
    plt.show()

def calculate_kurtosis_skewness(data_returns):
    kurtosis_values = data_returns.apply(kurtosis)
    skewness_values = data_returns.apply(skew)
    stats_df = pd.DataFrame({'Kurtosis': kurtosis_values, 'Skewness': skewness_values})
    print("Kurtosis and Skewness for each ETF:")
    print(stats_df)
    print("\nDescriptive Statistics:")
    print(data_returns.describe())
    return stats_df

def plot_correlation_matrix(data_returns):
    correlation_matrix = data_returns.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Daily Returns')
    plt.show()

def plot_daily_returns(data_returns, etfs):
    plt.figure(figsize=(10, 8))
    for etf in etfs:
        plt.plot(data_returns.index, data_returns[etf] * 100, label=f'{etf} Return')  # Convert returns to percentage
    plt.xlabel('Date')
    plt.ylabel('Scaled Daily Return (%)')
    plt.title('Daily Returns of ETFs')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(False)
    plt.show()

def plot_cumulative_returns(data_returns, etfs):
    cumulative_returns = (1 + data_returns).cumprod() - 1
    plt.figure(figsize=(14, 8))
    for ticker in etfs:
        plt.plot(cumulative_returns.index, cumulative_returns[ticker], label=f'{ticker} Cumulative Return')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Returns of Selected ETFs')
    plt.legend()
    plt.grid(False)
    plt.show()

def download_tbill_data(tbill_symbol, start_date, end_date):
    tbill_data = yf.download(tbill_symbol, start=start_date, end=end_date)
    tbill_data = tbill_data[['Adj Close']].rename(columns={'Adj Close': 'Risk Free Rate'})
    tbill_data['Risk Free Rate'] = tbill_data['Risk Free Rate'] / 100 / 252
    tbill_data.to_csv('risk_free_rate_data.csv', index=True)
    return tbill_data

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return - (p_returns - risk_free_rate) / p_std

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0, constraint_set=(0, 1)):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraint_set for asset in range(num_assets))
    result = sc.minimize(negative_sharpe_ratio, num_assets * [1. / num_assets], args=args,
                         method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_variance(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def minimize_variance(mean_returns, cov_matrix, constraint_set=(0, 1)):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraint_set for asset in range(num_assets))
    result = sc.minimize(portfolio_variance, num_assets * [1. / num_assets], args=args,
                         method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_return(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[0]

def efficient_optimization(mean_returns, cov_matrix, return_target, constraint_set=(0, 1)):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns, cov_matrix) - return_target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraint_set for asset in range(num_assets))
    eff_opt = sc.minimize(portfolio_variance, num_assets * [1. / num_assets], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return eff_opt

def display_portfolio_weights(weights, etfs):
    print("Portfolio Weights Allocation:")
    for i, weight in enumerate(weights):
        print(f"{etfs[i]}: {weight:.4f}")

def simulate_portfolio(weights, returns):
    portfolio_returns = returns.dot(weights)
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    return cumulative_returns

def plot_efficient_frontier(mean_returns, cov_matrix, etfs, data_returns):
    target_returns = np.linspace(0.01, 0.3, 100)
    efficient_portfolios = [efficient_optimization(mean_returns, cov_matrix, target) for target in target_returns]
    
    eff_returns = [portfolio_return(p['x'], mean_returns, cov_matrix) for p in efficient_portfolios]
    eff_volatilities = [portfolio_variance(p['x'], mean_returns, cov_matrix) for p in efficient_portfolios]
    
    plt.figure(figsize=(10, 6))
    plt.plot(eff_volatilities, eff_returns, label='Efficient Frontier', color='green')
    
    mean_returns_annualized = data_returns.mean() * 252
    std_devs_annualized = data_returns.std() * np.sqrt(252)
    plt.scatter(std_devs_annualized, mean_returns_annualized, color='blue', label='ETFs')
    
    offsets = {'TMSRX': (10, 0), 'CRAZX': (-30, 2), 'NBRVX': (-30, -10), 'VDIGX': (10, 0),
               'FBGRX': (10, 10), 'PRDGX': (-45, -2), 'BPTRX': (-30, 0), 'ACMVX': (10, 10)}

    for i, txt in enumerate(etfs):
        plt.annotate(txt, (std_devs_annualized[i], mean_returns_annualized[i]), 
                     xytext=offsets[txt], textcoords='offset points', fontsize=12)
    
    max_sr = max_sharpe_ratio(mean_returns, cov_matrix)
    max_sr_weights = max_sr.x
    portfolio_return_val, portfolio_std_dev = portfolio_performance(max_sr_weights, mean_returns, cov_matrix)
    plt.scatter(portfolio_std_dev, portfolio_return_val, color='red', label='Max Sharpe Ratio Portfolio', marker='x')

    plt.xlabel('Annualized Standard Deviation (Volatility)')
    plt.ylabel('Annualized Mean Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(False)
    plt.show()

def main():
    etfs = ['TMSRX', 'CRAZX', 'NBRVX', 'VDIGX', 'FBGRX', 'PRDGX', 'BPTRX', 'ACMVX']
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    tbill_symbol = '^IRX'
    
    data = download_data(etfs, start_date, end_date)
    data_returns = preprocess_data(data)
    mean_returns, cov_matrix = calculate_statistics(data_returns)
    
    plot_historical_prices(data, etfs)
    
    for etf in etfs:
        decompose_time_series(data, etf)
    
    perform_adf_test(data, etfs)
    plot_box_plots(data_returns)
    calculate_kurtosis_skewness(data_returns)
    plot_correlation_matrix(data_returns)
    plot_daily_returns(data_returns, etfs)
    plot_cumulative_returns(data_returns, etfs)
    
    tbill_data = download_tbill_data(tbill_symbol, start_date, end_date)
    
    max_sr = max_sharpe_ratio(mean_returns, cov_matrix)
    min_var = minimize_variance(mean_returns, cov_matrix)
    return_target = 0.2
    eff_opt = efficient_optimization(mean_returns, cov_matrix, return_target)
    
    display_portfolio_weights(max_sr.x, etfs)
    display_portfolio_weights(min_var.x, etfs)
    display_portfolio_weights(eff_opt.x, etfs)
    
    portfolio_cumulative_returns = simulate_portfolio(max_sr.x, data_returns)
    plt.figure(figsize=(12, 8))
    plt.plot(portfolio_cumulative_returns, label='Max Sharpe Ratio Portfolio')
    for ticker in etfs:
        plt.plot((1 + data_returns[ticker]).cumprod() - 1, label=ticker)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Returns of Portfolio vs Individual ETFs')
    plt.legend()
    plt.grid(False)
    plt.show()

    plot_efficient_frontier(mean_returns, cov_matrix, etfs, data_returns)

if __name__ == "__main__":
    main()
