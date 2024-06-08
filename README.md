# ETF Portfolio Optimization and Analysis

This project provides a comprehensive analysis and optimization of a portfolio of selected Exchange-Traded Funds (ETFs) using various financial and statistical techniques. The goal is to calculate statistics, optimize portfolio allocation, and visualize the results.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [ETFs Included](#etfs-included)
- [Optimization Techniques](#optimization-techniques)
- [Visualization](#visualization)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project downloads historical price data for a set of ETFs, preprocesses the data, calculates statistical metrics, and optimizes portfolio allocations to maximize the Sharpe ratio and minimize variance. The results are visualized using various plots.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/etf-portfolio-optimization.git
    cd etf-portfolio-optimization
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the main script to execute the entire analysis and optimization process:
    ```bash
    python main.py
    ```

2. The results, including plots and statistics, will be displayed and saved to the current directory.

## Features

- Download historical price data for selected ETFs.
- Preprocess the data by handling missing values and calculating daily returns.
- Perform time series decomposition.
- Conduct Augmented Dickey-Fuller tests to check for stationarity.
- Calculate and plot statistical metrics such as kurtosis, skewness, and correlation matrices.
- Optimize portfolio allocation to maximize the Sharpe ratio and minimize variance.
- Visualize the efficient frontier and cumulative returns.

## ETFs Included

The following ETFs are included in the analysis:
- TMSRX
- CRAZX
- NBRVX
- VDIGX
- FBGRX
- PRDGX
- BPTRX
- ACMVX

## Optimization Techniques

- **Maximize Sharpe Ratio**: Optimizes the portfolio to achieve the highest risk-adjusted return.
- **Minimize Variance**: Allocates weights to minimize the overall portfolio risk.
- **Efficient Frontier**: Plots the optimal portfolios for various target returns.

## Visualization

The project includes various plots to visualize the results:
- Historical prices of ETFs.
- Time series decomposition.
- Box plots of daily return distributions.
- Correlation matrix of daily returns.
- Daily returns of ETFs.
- Cumulative returns of the portfolio vs individual ETFs.
- Efficient frontier with annotated ETFs and optimized portfolio.

## Dependencies

The project requires the following packages:
- yfinance
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scipy
- tensorflow
- sklearn

Install the dependencies using:
```bash
pip install -r requirements.txt
License
This project is licensed under the MIT License. See the LICENSE file for details.
