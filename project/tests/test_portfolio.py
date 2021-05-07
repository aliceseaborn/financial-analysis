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


__author__ = "Alice Seaborn"

__version__ = "0.0.0"
__maintainer__ = "Alice Seaborn"
__email__ = "seaborn.dev@gmail.com"
__status__ = "Prototype"


test_stock_path = "project/tests/data/{}.csv"
test_figures_path = "project/tests/figures/marko"

start_date = "1900-01-01"
end_date = "1901-01-01"

risk_free_rate = 0.0165


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


def _instantiate_index():
    index_name = "^INDX"
    return Stock(index_name, "Generic Index", pd.read_csv(test_stock_path.format(index_name)))


class Test_PortfolioInitialization(object):
    
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


class Test_PortfolioIndividualRatesOfReturn(object):
    portfolio = _instantiate_portfolio()
    
    def test_individual_rate_of_return_values(self):
        result = self.portfolio.daily_individual_rates_of_return(start_date, end_date)
        assert round(result.iloc[1,1], 4) == 0.0583
        assert round(result.iloc[50,2], 4) == -0.045
        assert round(result.iloc[21,3], 4) == 0.0
        assert round(result.iloc[100,4], 4) == 0.0092
        assert round(result.iloc[245,5], 4) == 0.0852


class Test_PortfolioCombinedRateOfReturn(object):
    portfolio = _instantiate_portfolio()
    
    def test_combined_rate_of_return_values(self):
        individual_stocks = self.portfolio.daily_individual_rates_of_return(start_date, end_date)
        portfolio_combined = self.portfolio.daily_combined_rate_of_return(start_date, end_date)
        test_indexes = [1, 55, 132, 189, 245]
        truth = list()

        for i in test_indexes:
            result = float()
            for j in range(individual_stocks.shape[1]-1):
                result += individual_stocks.iloc[i,j+1] * self.portfolio.weights[j][0]

            assert portfolio_combined.iloc[i,1] == result


class Test_DailyExcessRatesOfReturn(object):
    portfolio = _instantiate_portfolio()
    
    def test_daily_excess_rates_of_return_values(self):
        result = self.portfolio.daily_excess_rates_of_return(risk_free_rate, start_date, end_date)
        assert round(result.iloc[0,1], 4) == -0.0001
        assert round(result.iloc[10,1], 4) == 0.0116
        assert round(result.iloc[57,1], 4) == -0.0307
        assert round(result.iloc[124,1], 4) == -0.0019
        assert round(result.iloc[189,1], 4) == 0.0198
        assert round(result.iloc[205,1], 4) == 0.0022
        
        
class Test_AverageDailyRateOfReturn(object):
    portfolio = _instantiate_portfolio()
    
    def test_average_daily_rate_of_return_values(self):
        result = self.portfolio.average_daily_rate_of_return(start_date, end_date)
        assert round(result, 4) == -0.4736
        
        
class Test_AverageAnnualRateOfReturn(object):
    portfolio = _instantiate_portfolio()
    
    def test_average_annual_rate_of_return_value(self):
        result = self.portfolio.average_annual_rate_of_return(start_date, end_date)
        assert round(result, 4) == -118.4106
        
        
class Test_IndividualVariances(object):
    portfolio = _instantiate_portfolio()
    
    def test_individual_variance_values(self):
        result = self.portfolio.individual_variances(start_date, end_date)
        assert round(result["TS Variance"][0], 4) == 0.3667
        assert round(result["TSB Variance"][0], 4) == 0.8901
        assert round(result["TSC Variance"][0], 4) == 0.0152
        assert round(result["TSD Variance"][0], 4) == 0.0152
        assert round(result["TSE Variance"][0], 4) == 1.8330
        

class Test_CovarianceMatrix(object):
    portfolio = _instantiate_portfolio()
    
    def test_covariance_matrix_diagonal_equals_stock_variance(self):
        result = self.portfolio.covariance_matrix(start_date, end_date)
        assert round(self.portfolio.stocks[0].daily_rate_of_return("l", start_date, end_date).iloc[:,1].var(), 4) == round(result.iloc[0,0], 4)
        assert round(self.portfolio.stocks[1].daily_rate_of_return("l", start_date, end_date).iloc[:,1].var(), 4) == round(result.iloc[1,1], 4)
        assert round(self.portfolio.stocks[2].daily_rate_of_return("l", start_date, end_date).iloc[:,1].var(), 4) == round(result.iloc[2,2], 4)
        assert round(self.portfolio.stocks[3].daily_rate_of_return("l", start_date, end_date).iloc[:,1].var(), 4) == round(result.iloc[3,3], 4)
        assert round(self.portfolio.stocks[4].daily_rate_of_return("l", start_date, end_date).iloc[:,1].var(), 4) == round(result.iloc[4,4], 4)
        
    def test_covariance_matrix_non_diagonal_values(self):
        result = self.portfolio.covariance_matrix(start_date, end_date)
        assert round(result.iloc[1, 2], 6) == -1.5e-05
        assert round(result.iloc[4, 0], 6) == 0.000157
        assert round(result.iloc[3, 4], 6) == -8.9e-05


class Test_Variance(object):
    portfolio = _instantiate_portfolio()
    
    def test_variance_value(self):
        result = self.portfolio.variance(start_date, end_date)
        assert result == 0.05996553983605832


class Test_AnnualStandardDeviation(object):
    portfolio = _instantiate_portfolio()
    
    def test_annual_standard_deviation_value(self):
        result = self.portfolio.annual_standard_deviation(start_date, end_date)
        assert result == 6.109929240296508


class Test_Volatility(object):
    portfolio = _instantiate_portfolio()
    
    def test_annual_volatility_value(self):
        result = self.portfolio.volatility(start_date, end_date)
        assert result == 24.487862266040768


class Test_Correlation(object):
    portfolio = _instantiate_portfolio()
    
    def test_correlation_matrix_diagonal(self):
        result = self.portfolio.correlation_matrix(start_date, end_date)
        assert result.iloc[0,0] == 1.0
        assert result.iloc[1,1] == 1.0
        assert result.iloc[2,2] == 1.0
        assert result.iloc[3,3] == 1.0
        assert result.iloc[4,4] == 1.0
        
    def test_correlation_matrix_upper_lower_bounds(self):
        result = self.portfolio.correlation_matrix(start_date, end_date)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                assert result.iloc[i, j] <= 1.0
                assert result.iloc[i, j] >= -1.0
                
    def test_correlation_matrix_values(self):
        result = self.portfolio.correlation_matrix(start_date, end_date)
        assert result.iloc[0,1] == 0.10855111259528193
        assert result.iloc[1,3] == -0.03099991251268967
        assert result.iloc[1,2] == -0.03099991251268967
        assert result.iloc[3,4] == -0.1306416315985334
        assert result.iloc[4,2] == -0.1306416315985334


class Test_IndividualBetas(object):
    portfolio = _instantiate_portfolio()
    index = _instantiate_index()
    
    def test_individual_beta_values(self):
        result = self.portfolio.individual_betas(self.index, start_date, end_date)
        assert result.iloc[0,0] == self.portfolio.stocks[0].beta(self.index, start_date, end_date)
        assert result.iloc[0,1] == self.portfolio.stocks[1].beta(self.index, start_date, end_date)
        assert result.iloc[0,2] == self.portfolio.stocks[2].beta(self.index, start_date, end_date)
        assert result.iloc[0,3] == self.portfolio.stocks[3].beta(self.index, start_date, end_date)


class Test_Beta(object):
    portfolio = _instantiate_portfolio()
    index = _instantiate_index()
    
    def test_beta_value(self):
        result = self.portfolio.beta(self.index, start_date, end_date)
        assert result == 0.4315856457690673


class Test_SystematicRisk(object):
    portfolio = _instantiate_portfolio()
    index = _instantiate_index()
    
    def test_systematic_risk_value(self):
        result = self.portfolio.systematic_risk(self.index, start_date, end_date)
        assert result == 1.6088583059946866


class Test_IdiosyncraticRisk(object):
    portfolio = _instantiate_portfolio()
    index = _instantiate_index()
    
    def test_idiosyncratic_risk_value(self):
        result = self.portfolio.idiosyncratic_risk(self.index, start_date, end_date)
        assert result == 1.6086726668290694


class Test_SharpeRatio(object):
    portfolio = _instantiate_portfolio()
    
    def test_sharpe_ratio_value(self):
        result = self.portfolio.sharpe_ratio(risk_free_rate, start_date, end_date)
        assert result == -19.382720622879468


class Test_MarkowitzEfficientFrontier(object):
    portfolio = _instantiate_portfolio()
    result = portfolio.markowitz_efficient_frontier(100, risk_free_rate, test_figures_path,
                                                    start_date, end_date)
    
    def test_marko_length(self):
        assert len(self.result) == 100
        
    def test_marko_weights_length(self):
        assert len(self.result[0]["Weights"]) == len(self.portfolio.stocks)
        
    def test_marko_weights_bounds(self):
        for i in range(len(self.result)):
            weights = self.result[i]["Weights"]
            for weight in weights:
                assert weight <= 1.0
                assert weight >= 0.0
                
    def test_marko_weights_sum(self):
        for i in range(len(self.result)):
            assert round(sum(self.result[i]["Weights"])[0], 3) == 1.000
                



