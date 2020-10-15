#!/usr/bin/env python
"""Object representing a single tradable equity.
"""

import pandas as pd
import numpy as np

__author__ = "Austin Dial, Alice Seaborn"

__version__ = "0.0.0"
__maintainer__ = "Alice Seaborn"
__email__ = "seaborn.archipelago@gmail.com"
__status__ = "Prototype"



class Stock(object):
    
    def __init__(self, ticker, name, history):
        self.ticker = ticker
        self.name = name
        self.history = history

    
    def daily_rate_of_return(self, mode='single', start_date=None, end_date=None):
        """Series of daily rates of return in either simple ('s') or logarithmic ('l')
        mode between the given dates. Date format is "yyyy-mm-dd". Returns are not
        expressed as a percentage.
        
            DAILY LOG ROR = LOG( CLOSE / CLOSE.SHIFT(1) )
            DAILY SIMPLE ROR = ( CLOSE / CLOSE.SHIFT(1) ) - 1
                s.t. ROR - rate of return
        """
        
        if not start_date:
            start_date = self.history.iloc[0,0]
        if not end_date:
            end_date = self.history.iloc[len(self.history)-1,0]
        
        selection = self.history.query(f"Date >= '{start_date}' and Date " \
                                      "<= '{end_date}'").reset_index(drop=True)
        dates = selection.iloc[:,0]
        
        if mode == "simple" or mode == "s":
            returns = ((selection.iloc[:,5] / selection.iloc[:,5].\
                        shift(1)) - 1)
        elif mode == "logarithmic" or mode == "l":
            returns = np.log(selection.iloc[:,5] / selection.iloc[:,5].\
                             shift(1))
        else:
            raise ValueError(f"Mode: {mode} is unacceptable.")
        
        stock_daily_ror = pd.concat([dates, returns], axis=1)
        stock_daily_ror = stock_daily_ror.rename(columns={"Date": "Date", \
                                                          "Adj Close": self.ticker + " ROR"})
        
        return stock_daily_ror

    
    def average_daily_rate_of_return(self, mode='s', start_date=None, end_date=None):
        """Average daily simple or logarithmic rate of return between the given
        dates. Average return is expressed as a percentage. Date format is 
        "yyyy-mm-dd".

            AVERAGE DAILY ROR = MEAN( DAILY ROR ) * 100
                s.t. ROR - rate of return
        """

        daily_ror = self.daily_rate_of_return(mode, start_date, end_date)
        average_daily_ror = daily_ror.iloc[:,1].mean() * 100

        return average_daily_ror
    
    
    def average_annual_rate_of_return(self, mode='simple', start_date=None, end_date=None):
        """Average annual simple or logarithmic rate of return between the given
        dates. Date format is "yyyy-mm-dd".

            AVERAGE ANNUAL ROR = AVERAGE DAILY ROR * 250
                s.t. ROR - rate of return
        """

        average_daily_ror = self.average_daily_rate_of_return(mode, start_date, end_date)
        average_annual_ror = average_daily_ror * 250

        return average_annual_ror
    
    
    def annual_volatility(self, start_date=None, end_date=None):
        """Annual standard deviation of the stock returns over the specified period.
        The standard deviation is expressed as a percentage. Date format is
        "yyyy-mm-dd".

            ST. DEV = STD( DAILY STOCK ROR ) * 250 * 100
                s.t. ROR - rate of return
        """

        stock_ror = self.daily_rate_of_return('simple', start_date, end_date)
        stock_std = stock_ror.iloc[:,1].std() * 250 * 100

        return stock_std
    
    
    def annual_variance(self, start_date=None, end_date=None):
        """Annual variance of the stock returns over the specified period. The
        variance is not expressed as a percentage. Date format is "yyyy-mm-dd".

            VARIANCE = VAR( DAILY STOCK ROR ) * 250
                s.t. ROR - rate of return
        """

        stock_ror = self.daily_rate_of_return('simple', start_date, end_date)
        stock_var = stock_ror.iloc[:,1].var() * 250

        return stock_var


    def beta(self, index, start_date=None, end_date=None):
        """The beta for the stock against the market index, provided as
        a stock object. Date format is "yyyy-mm-dd".

            BETA = COV( STOCK RETURNS, INDEX RETURNS ) / VAR( INDEX )
        """

        stock_ret = self.daily_rate_of_return('logarithmic', "2020-01-01", "2020-10-02")
        index_ret = index.daily_rate_of_return('logarithmic', "2020-01-01", "2020-10-02")

        covariance = np.cov(stock_ret.iloc[:,1][1:], index_ret.iloc[:,1][1:])[0,0]
        variance = np.var(index_ret.iloc[:,1])

        beta = covariance / variance

        return beta
