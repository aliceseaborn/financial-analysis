#!/usr/bin/env python
"""Functions for analyzing individual tradable equities.

Eventually, these functions will become member functions of domain objects.
"""

import pandas as pd
import numpy as np

__author__ = "Austin Dial, Alice Seaborn"

__version__ = "0.0.0"
__maintainer__ = "Alice Seaborn"
__email__ = "adial@mail.bradley.edu"
__status__ = "Prototype"



# ------------------------- RATES OF RETURN ------------------------- #

def daily_stock_rate_of_return(Stock, mode='s', start_date=None, end_date=None):
    """Series of daily simple or logarithmic rates of return between the given
    dates. Returns are not expressed as a percentage.
    
        DAILY LOG ROR = LOG( CLOSE / CLOSE.SHIFT(1) )
        DAILY SIMPLE ROR = ( CLOSE / CLOSE.SHIFT(1) ) - 1
            s.t. ROR - rate of return

    Parameters
    ----------
    Stock : Stock object.
        The stock for analysis.
    start_date : String.
        Sets the start of the stock analysis. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the stock analysis. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """

    if not start_date:
        start_date = Stock.history.iloc[0,0]
    if not end_date:
        end_date = Stock.history.iloc[len(Stock.history)-1,0]
        
    selection = Stock.history.query(f"Date >= '{start_date}' and Date " \
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
                                                      "Adj Close": Stock.ticker + " ROR"})
    
    return stock_daily_ror 


def average_daily_stock_rate_of_return(Stock, mode='s', start_date=None, end_date=None):
    """Average daily simple or logarithmic rate of return between the given
    dates. Average return is expressed as a percentage.
    
        AVERAGE DAILY ROR = MEAN( DAILY ROR ) * 100
            s.t. ROR - rate of return

    Parameters
    ----------
    Stock : Stock object.
        The stock for analysis.
    start_date : String.
        Sets the start of the stock analysis. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the stock analysis. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """

    daily_ror = daily_stock_rate_of_return(Stock, mode, start_date, end_date)
    average_daily_ror = daily_ror.iloc[:,1].mean() * 100
    
    return average_daily_ror


def average_annual_stock_rate_of_return(Stock, mode='s', start_date=None, end_date=None):
    """Average annual simple or logarithmic rate of return between the given
    dates.
    
        AVERAGE ANNUAL ROR = AVERAGE DAILY ROR * 250
            s.t. ROR - rate of return

    Parameters
    ----------
    Stock : Stock object.
        The stock for analysis.
    start_date : String.
        Sets the start of the stock analysis. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the stock analysis. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """

    average_daily_ror = average_daily_stock_rate_of_return(Stock, mode, start_date, \
                                                     end_date)
    average_annual_ror = average_daily_ror * 250
    
    return average_annual_ror


# ------------------------- VOLATILITY ------------------------- #

def stock_annual_volatility(Stock, start_date=None, end_date=None):
    """Annual standard deviation of the stock returns over the specified period.
    The standard deviation is expressed as a percentage.

        ST. DEV = STD( DAILY STOCK ROR ) * 250 * 100
            s.t. ROR - rate of return

    Parameters
    ----------
    Stock : Stock object.
        The stock for analysis.
    start_date : String.
        Sets the start of the stock analysis. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the stock analysis. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """

    stock_ror = daily_stock_rate_of_return(Stock, 's', start_date, end_date)
    stock_std = stock_ror.iloc[:,1].std() * 250 * 100
    
    stock_std = pd.DataFrame(data=[stock_std], 
                             columns=[Stock.ticker + " Standard Deviation"])
    
    return stock_std


# ------------------------- VARIANCE ------------------------- #

def stock_annual_variance(Stock, start_date=None, end_date=None):
    """Annual variance of the stock returns over the specified period.
    The variance is not expressed as a percentage.
    
        VARIANCE = VAR( DAILY STOCK ROR ) * 250
            s.t. ROR - rate of return
    
    Parameters
    ----------
    Stock : Stock object.
        The stock for analysis.
    start_date : String.
        Sets the start of the stock analysis. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the stock analysis. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """

    stock_ror = daily_stock_rate_of_return(Stock, 's', start_date, end_date)
    stock_var = stock_ror.iloc[:,1].var() * 250
    
    stock_var = pd.DataFrame(data=[stock_var], 
                             columns=[Stock.ticker + " Variance"])
    
    return stock_var


# ------------------------- STOCK BETA ------------------------- #

def stock_beta(Stock, Index, start_date=None, end_date=None):
    """The beta for the Stock against the provided market index.
    
        BETA = COV( STOCK RETURNS, INDEX RETURNS ) / VAR( INDEX )
    
    Parameters
    ----------
    Portfolio : Stock object.
        A stock for correlation analysis.
    Index : Stock object.
        A stock representing the provided composite market index.
    start_date : String.
        Sets the start of the correlation study. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the correlation study. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """

    stock_ret = daily_stock_rate_of_return(stock, 'l', start_date, end_date)
    index_ret = daily_stock_rate_of_return(Index, 'l', start_date, end_date)

    covariance = np.cov(stock_ret.iloc[:,1], index_ret.iloc[:,1])[0,0]
    variance = np.var(index_ret.iloc[:,1])

    beta = covariance / variance
    
    return beta


