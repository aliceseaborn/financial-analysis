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



test_stock_path = "project/tests/data/TS.csv"
test_index_path = "project/tests/data/^INDX.csv"



class TestInitialization(object):
    
    def test_stock_type(self):
        
        result = Stock('TS', 'Test Stock', pd.read_csv(test_stock_path))
        assert isinstance(result, Stock) == True
    
    
    def test_stock_attributes(self):
        
        result = Stock('TS', 'Test Stock', pd.read_csv(test_stock_path))
        assert result.ticker == "TS"
        assert result.name == "Test Stock"
        assert result.history.all().all() == pd.read_csv(test_stock_path).all().all()
        

    def test_type_error_handling(self):
        
        result = Stock(5.0, 5.0, pd.read_csv(test_stock_path))
        assert result.name == "5.0"
        assert result.ticker == "5.0"



class Test_AnnualizationFactor(object):
    
    stock = Stock('TS', 'Test Stock', pd.read_csv(test_stock_path))
    
    def test_less_than_year_default_value(self):
        
        results = self.stock.annualization_factor("1900-01-01", "1900-06-01")
        assert results == 250
    
    
    def test_one_year_value(self):
        
        result = self.stock.annualization_factor("1900-01-01", "1901-01-01")
        assert result == 246.0



class Test_SimpleDailyRateOfReturn(object):
    
    stock = Stock('TS', 'Test Stock', pd.read_csv(test_stock_path))
    
    def test_result_length_by_date_range(self):
        
        results = self.stock.daily_rate_of_return('s')
        assert len(results) == 743
        
        results = self.stock.daily_rate_of_return('s', '1900-01-02', '1900-01-03')
        assert len(results) == 2
    
    
    def test_rate_of_return_value(self):
        
        result = self.stock.daily_rate_of_return('s', '1900-01-01', '1900-01-10').iloc[:,1].values[1]
        assert round(result, 4) == 0.0600
    


class Test_LogarithmicDailyRateOfReturn(object):
    
    stock = Stock('TS', 'Test Stock', pd.read_csv(test_stock_path))
    
    def test_result_length_by_date_range(self):
        
        results = self.stock.daily_rate_of_return('l')
        assert len(results) == 743
        
        results = self.stock.daily_rate_of_return('l', '1900-01-02', '1900-01-03')
        assert len(results) == 2
    
    
    def test_rate_of_return_value(self):
        
        result = self.stock.daily_rate_of_return('l', '1900-01-01', '1900-01-10').iloc[:,1].values[1]
        assert round(result, 4) == 0.0583
        


class Test_AverageSimpleDailyRateOfReturn(object):
    
    stock = Stock('TS', 'Test Stock', pd.read_csv(test_stock_path))
    
    def test_average_rate_of_return_value(self):
        
        result = self.stock.average_daily_rate_of_return(mode='s')
        assert round(result, 4) == -0.0006
        

        
class Test_AverageLogarithmicDailyRateOfReturn(object):
    
    stock = Stock('TS', 'Test Stock', pd.read_csv(test_stock_path))
    
    def test_average_rate_of_return_value(self):
        
        result = self.stock.average_daily_rate_of_return(mode='l')
        assert round(result, 4) == -0.0013



class Test_AnnualRatesOfReturn(object):
    
    stock = Stock('TS', 'Test Stock', pd.read_csv(test_stock_path))
    
    def test_rate_of_return_value(self):
        
        result = self.stock.annual_rates_of_return(["1900", "1901"]).values[0][1]
        assert round(result, 4) == -71.0



class Test_AverageAnnualRatesOfReturn(object):
    
    stock = Stock('TS', 'Test Stock', pd.read_csv(test_stock_path))
    
    def test_rate_of_return_value(self):
        
        result = self.stock.average_annual_rate_of_return(["1900", "1901", "1902"])
        assert round(result, 4) == 55.24



class Test_AnnualVolatility(object):
    
    stock = Stock('TS', 'Test Stock', pd.read_csv(test_stock_path))
    
    def test_annual_volatility_value(self):
        
        result = self.stock.annual_volatility("1900-01-01", "1902-12-31")
        assert round(result, 4) == 9.7363



class Test_AnnualVariance(object):
    
    stock = Stock('TS', 'Test Stock', pd.read_csv(test_stock_path))
    
    def test_annual_variance_value(self):
        
        result = self.stock.annual_variance("1900-01-01", "1902-12-31")
        assert round(result, 4) == 0.3826



class Test_Beta(object):
    
    stock = Stock('TS', 'Test Stock', pd.read_csv(test_stock_path))
    index = Stock("^INDX", "Generic Index", pd.read_csv(test_index_path))
    
    def test_beta(self):
        
        result = self.stock.beta(self.index, "1900-01-01", "1902-12-31")
        assert round(result, 4) == 1.0918




