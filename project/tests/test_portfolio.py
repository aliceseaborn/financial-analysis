#!/usr/bin/env python
"""Object representing a single tradable equity.
"""

# Handle parent directory
import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
print(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# Test subject
from domain.Stock import Stock
from domain.Portfolio import Portfolio

# Pytest tools
import pytest

# Testing dependencies
import pandas as pd
import numpy as np


__author__ = "Austin Dial, Alice Seaborn"

__version__ = "0.0.0"
__maintainer__ = "Alice Seaborn"
__email__ = "seaborn.archipelago@gmail.com"
__status__ = "Prototype"



test_stock_path = "project/tests/data/{}.csv"



def _instantiate_portfolio():
    
    weights = np.array([ [0.40], [0.25], [0.15], [0.10], [0.10] ])
    tickers = ["TS", "TSB", "TSC", "TSD", "TSE"]
    companies = ["TS", "TS Beta", "TS Charlie", "TS Delta", "TS Echo"]
    entries = dict(zip(tickers, companies))

    stocks = list()
    for ticker, company in entries.items():
        history = pd.read_csv(test_stock_path.format(ticker))
        stocks.append(Stock(ticker, company, history))

    return Portfolio("Test Portfolio", stocks, weights)



class TestPortfolioInitialization(object):
    
    def test_portfolio_type(self):
        
        weights = np.array([ [0.40], [0.25], [0.15], [0.10], [0.10] ])
        tickers = ["TS", "TSB", "TSC", "TSD", "TSE"]
        companies = ["TS", "TS Beta", "TS Charlie", "TS Delta", "TS Echo"]
        entries = dict(zip(tickers, companies))

        stocks = list()
        for ticker, company in entries.items():
            history = pd.read_csv(test_stock_path.format(ticker))
            stocks.append(Stock(ticker, company, history))

        result = Portfolio("Test Portfolio", stocks, weights)
        
        assert isinstance(result, Portfolio) == True
    
    
    def test_portfolio_attributes(self):
        
        result = _instantiate_portfolio()
        assert result.name == "Test Portfolio"
        assert result.stocks[0].name == "TS"



class TestPortfolioIndividualRatesOfReturn(object):
    
    portfolio = _instantiate_portfolio()
    
    def test_individual_ror_values(self):
        
        result = self.portfolio.daily_individual_rates_of_return("1900-01-01", "1901-01-01")
        assert round(result.iloc[1,1], 4) == 0.0583
        assert round(result.iloc[50,2], 4) == -0.045
        assert round(result.iloc[21,3], 4) == 0.0
        assert round(result.iloc[100,4], 4) == 0.0092
        assert round(result.iloc[245,5], 4) == 0.0852



class TestPortfolioCombinedRateOfReturn(object):
    
    portfolio = _instantiate_portfolio()
    
    def test_combined_ror_values(self):
        
        individual_stocks = self.portfolio.daily_individual_rates_of_return("1900-01-01", "1901-01-01")
        portfolio_combined = self.portfolio.daily_combined_rate_of_return("1900-01-01", "1901-01-01")
        
        test_indexes = [1, 55, 132, 189, 245]
        truth = list()

        for i in test_indexes:
            result = float()
            for j in range(individual_stocks.shape[1]-1):
                result += individual_stocks.iloc[i,j+1] * self.portfolio.weights[j][0]

            assert portfolio_combined.iloc[i,1] == result



class TestPortfolioIndividualRatesOfReturn(object):
    
    portfolio = _instantiate_portfolio()
    
    def test_individual_ror_values(self):
        
        result = self.portfolio.daily_individual_rates_of_return("1900-01-01", "1901-01-01")
        assert round(result.iloc[1,1], 4) == 0.0583
        assert round(result.iloc[50,2], 4) == -0.045
        assert round(result.iloc[21,3], 4) == 0.0
        assert round(result.iloc[100,4], 4) == 0.0092
        assert round(result.iloc[245,5], 4) == 0.0852






