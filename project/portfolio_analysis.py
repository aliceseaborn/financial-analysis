#!/usr/bin/env python
"""Functions for analyzing portfolios of tradable equities.

Eventually, these functions will become member functions of domain objects.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable

import pandas as pd
import numpy as np
from numpy.random import random

from stock_analysis import daily_stock_rate_of_return, stock_annual_variance
from domain.Portfolio import Portfolio


__author__ = "Austin Dial, Alice Seaborn"

__version__ = "0.0.0"
__maintainer__ = "Alice Seaborn"
__email__ = "adial@mail.bradley.edu"
__status__ = "Prototype"



# ------------------------- RATES OF RETURN ------------------------- #

def daily_portfolio_individual_rates_of_return(portfolio, start_date=None, end_date=None):
    """Calculates the daily return histories for each individual stock
    in the portfolio.
    
        FOR STOCK IN PORTFOLIO:
            STOCK ROR = LOG( CLOSE / CLOSE.SHIFT(1) )
    
    Parameters
    ----------
    portfolio : Portfolio object.
        The portfolio for analysis.
    start_date : String.
        Sets the start of the portfolio analysis. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the portfolio analysis. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """
    
    dataframes = list([])
    for stock in portfolio.stocks:
        returns = daily_stock_rate_of_return(stock, 'l', start_date, end_date)
        df = pd.DataFrame(data=returns.iloc[:,1].to_numpy(), 
                          columns=[stock.ticker + ' Returns'])
        dataframes.append(df)

    dataframes.insert(0, pd.DataFrame(data=returns['Date'], 
                                      columns=['Date']))
    
    return pd.concat(dataframes, axis=1)


def daily_portfolio_combined_rate_of_return(portfolio, start_date=None, end_date=None):
    """Calculates the daily return history for portfolio between the
    given dates. 

    Parameters
    ----------
    portfolio : Portfolio object.
        The portfolio for analysis.
    start_date : String.
        Sets the start of the portfolio analysis. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the portfolio analysis. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """
    
    # Calculate the daily rates of return for each stock
    indiv_ret = daily_portfolio_individual_rates_of_return(portfolio, start_date, end_date)

    # Distribute weights to returns
    weighted_indiv_ret = pd.DataFrame()
    for i in range(len(indiv_ret.columns)-1):
        weighted_indiv_ret[indiv_ret.columns[i+1]] = indiv_ret[indiv_ret.columns[i+1]].astype(float) * portfolio.weights[i][0]

    # Calculate the portfolio's daily rate of return
    total_returns = pd.Series(data=weighted_indiv_ret.sum(axis=1), 
                              name=portfolio.name + " Returns")
    dates = pd.Series(data=indiv_ret['Date'], name="Date")
    port_ret = pd.concat([dates, total_returns], axis=1)
    
    return port_ret


def daily_portfolio_excess_rates_of_return(portfolio, risk_free_rate, start_date=None, end_date=None):
    """Calculates the combined portfolio returns after considering the 
    risk free rate of return. This represents the excess returns the
    portfolio yields as a result of taking risk.
    
        EXCESS PORTFOLIO RETURNS = DAILY PORTFOLIO COMBINED 
            RATE OF RETURNS - ( RISK FREE RATE OF RETURNS / 250 )
    
    Parameters
    ----------
    portfolio : Portfolio object.
        The portfolio for analysis.
    risk_free_rate : Float.
        The annual risk free rate of return.
    start_date : String.
        Sets the start of the portfolio analysis. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the portfolio analysis. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """
    
    excess_daily_returns = pd.DataFrame()
    daily_returns = daily_portfolio_combined_rate_of_return(portfolio, start_date, end_date)
    excess_daily_returns['Date'] = daily_returns.iloc[:,0]
    excess_daily_returns[portfolio.name + ' Excess Daily Returns'] = daily_returns.iloc[:,1] - ( risk_free_rate / 250 )
    
    return excess_daily_returns


def average_daily_portfolio_rate_of_return(portfolio, start_date=None, end_date=None):
    """Average daily rate of return for the given portfolio between the 
    given dates. Average return is expressed as a percentage.
    
        AVE. DAILY PORT ROR = MEAN( PORT ROR ) * 100

    Parameters
    ----------
    portfolio : Portfolio object.
        The portfolio for analysis.
    start_date : String.
        Sets the start of the portfolio analysis. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the portfolio analysis. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """

    portfolio_ror = daily_portfolio_combined_rate_of_return(portfolio, start_date, end_date)
    average_daily_portfolio_ror = portfolio_ror[portfolio.name + ' Returns'].mean() * 100

    ave_daily_ret = pd.DataFrame(data=[average_daily_portfolio_ror], 
                                 columns=[portfolio.name + ' Average Daily Return'])

    return ave_daily_ret


def average_annual_portfolio_rate_of_return(portfolio, start_date=None, end_date=None):
    """Average annual rate of return for the given portfolio between the 
    given dates.

    Parameters
    ----------
    portfolio : Portfolio object.
        The portfolio for analysis.
    start_date : String.
        Sets the start of the portfolio analysis. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the portfolio analysis. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """

    ave_daily_ret = average_daily_portfolio_rate_of_return(portfolio, start_date, end_date)
    ave_annual_return = pd.DataFrame(data=[ave_daily_ret.values[0][0] * 250],
                                     columns=[portfolio.name + ' Average Annual Return'])

    return ave_annual_return


# ------------------------- INDIVIDUAL VARIANCES ------------------------- #

def portfolio_individual_variances(portfolio, start_date=None, end_date=None):
    """Variances for each stock in the portfolio.

    Parameters
    ----------
    portfolio : Portfolio object.
        A portfolio whose stock variances will be analyzed.
    start_date : String.
        Sets the start of the variance study. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the variance study. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """
    
    data = dict()
    for stock in portfolio.stocks:
        data[stock.ticker + ' Variance'] = [stock_annual_variance(stock, start_date, end_date).values[0][0]]

    individual_variances = pd.DataFrame(data)
    
    return individual_variances


# ------------------------- PORTFOLIO VARIANCE ------------------------- #

def portfolio_variance(portfolio, start_date=None, end_date=None):
    """The annual variance of the portfolio.
    
        VARIANCE = W^T * COVARIANCE * W
            s.t. W = Weights
    
    Parameters
    ----------
    portfolio : Portfolio object.
        A portfolio whose variance will be analyzed.
    start_date : String.
        Sets the start of the variance study. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the variance study. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """
    
    cov_matrix = covariance_matrix(portfolio, start_date, end_date)
    weights = portfolio.weights

    variance = np.matmul( np.matmul( weights.T, cov_matrix.to_numpy() ), 
                         weights )[0,0] * 100
    
    port_variance = pd.DataFrame(data=[variance], columns=[portfolio.name \
                                                           + ' Variance'])
    
    return port_variance


# ------------------------- PORTFOLIO VARIANCE ------------------------- #

def portfolio_annual_standard_deviation(portfolio, start_date=None, end_date=None):
    """The annual standard deviation of the portfolio returns.
    
        STANDARD DEVIATION = STD( PORTFOLIO DAILY RETURNS ) * 250
    Parameters
    ----------
    portfolio : Portfolio object.
        A portfolio whose standard deviation will be analyzed.
    start_date : String.
        Sets the start of the standard deviation study. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the standard deviation study. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """
    
    returns = daily_portfolio_combined_rate_of_return(portfolio, start_date, end_date)
    returns_std = returns.iloc[:,1].std() * 250
    
    port_standard_deviation = pd.DataFrame(data=[returns_std], columns=[portfolio.name \
                                                           + ' Annual Standard Deviation'])
    
    return port_standard_deviation


# ------------------------- PORTFOLIO VOLATILITY ------------------------- #

def portfolio_volatility(portfolio, start_date=None, end_date=None):
    """The annual volatility of the portfolio expressed as a percentage.
    
        VOLATILITY = SQRT( W^T * COVARIANCE * W )
            s.t. W = Weights
    
    Parameters
    ----------
    portfolio : Portfolio object.
        A portfolio whose volatility will be analyzed.
    start_date : String.
        Sets the start of the volatility study. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the volatility study. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """
    
    variance = portfolio_variance(portfolio, start_date, end_date).values[0, 0]
    volatility = np.sqrt( variance ) * 100
    
    port_volatility = pd.DataFrame(data=[volatility], 
                                   columns=[portfolio.name + ' Volatility'])
    
    return port_volatility


# ------------------------- CAVARIANCE MATRIX ------------------------- #

def covariance_matrix(portfolio, start_date=None, end_date=None):
    """Covariance matrix of the provided stocks.
    
    The covariance is calculated by calling .cov() against the individual
    daily rates of return for the portfolio's stocks.
    
    Parameters
    ----------
    portfolio : Portfolio object.
        A portfolio for covariance analysis.
    start_date : String.
        Sets the start of the covariance study. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the covariance study. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """

    indiv_ret = daily_portfolio_individual_rates_of_return(portfolio, start_date, end_date)
    cov_matrix = indiv_ret.cov()
    
    return cov_matrix


# ------------------------- CORRELATION MATRIX ------------------------- #

def correlation_matrix(portfolio, start_date=None, end_date=None):
    """Correlation matrix of the provided stocks.
    
    The correlation is calculated by calling .corr() against the individual
    daily rates of return for the portfolio's stocks.
    
    Parameters
    ----------
    portfolio : Portfolio object.
        A portfolio for correlation analysis.
    start_date : String.
        Sets the start of the correlation study. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the correlation study. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """

    indiv_ret = daily_portfolio_individual_rates_of_return(portfolio, start_date, end_date)
    corr_matrix = indiv_ret.corr()
    
    return corr_matrix


# ------------------------- PORTFOLIO INDIVIDUAL BETAS ------------------------- #

def portfolio_individual_betas(portfolio, Index, start_date=None, end_date=None):
    """The betas for each stock in the Portfolio against the provided market index.
    
        BETA = COV( STOCK RETURNS, INDEX RETURNS ) / VAR( INDEX RETURNS )
    
    Parameters
    ----------
    portfolio : Portfolio object.
        A portfolio for correlation analysis.
    Index : Stock object.
        A stock representing the provided composite market index.
    start_date : String.
        Sets the start of the beta study. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the beta study. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """

    portfolio_ret = daily_portfolio_combined_rate_of_return(portfolio, start_date, end_date)
    index_ret = daily_stock_rate_of_return(Index, 'l', start_date, end_date)

    covariance = np.cov(portfolio_ret.iloc[:,1], index_ret.iloc[:,1])[0,0]
    variance = np.var(index_ret.iloc[:,1])

    beta = covariance / variance
    
    portfolio_beta = pd.DataFrame(data=[beta], 
                                   columns=[portfolio.name + ' Beta'])
    
    return portfolio_beta


# ------------------------- PORTFOLIO BETA ------------------------- #

def portfolio_beta(portfolio, Index, start_date=None, end_date=None):
    """The beta for the Portfolio against the provided market index.
    
        BETA = COV( PORTFOLIO RETURNS, INDEX RETURNS ) / VAR( INDEX )
    
    Parameters
    ----------
    portfolio : Portfolio object.
        A portfolio for correlation analysis.
    Index : Stock object.
        A stock representing the provided composite market index.
    start_date : String.
        Sets the start of the beta study. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the beta study. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """

    portfolio_ret = daily_portfolio_combined_rate_of_return(portfolio, start_date, end_date)
    index_ret = daily_stock_rate_of_return(Index, 'l', start_date, end_date)

    covariance = np.cov(portfolio_ret.iloc[:,1], index_ret.iloc[:,1])[0,0]
    variance = np.var(index_ret.iloc[:,1])

    beta = covariance / variance
    
    portfolio_beta = pd.DataFrame(data=[beta], 
                                   columns=[portfolio.name + ' Beta'])
    
    return portfolio_beta


# ------------------------- SYSTEMATIC RISK ------------------------- #

def systematic_risk(portfolio, Index, start_date=None, end_date=None):
    """Estimate of the Portfolio's systematic (undiversifiable) risk.
    Risk is expressed as a percentage.
    
        SYSTEMATIC RISK = PORTFOLIO BETA * STD( INDEX ) * 100
    
    Parameters
    ----------
    portfolio : Portfolio object.
        A portfolio for correlation analysis.
    Index : Stock object.
        A stock representing the provided composite market index.
    start_date : String.
        Sets the start of the riskbeta study. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the riskbeta study. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """
    
    beta = portfolio_beta(portfolio, Index, start_date, end_date).values[0,0]

    index_ret = daily_stock_rate_of_return(Index, 'l', start_date, end_date)

    sys_risk = beta * index_ret.std().values[0]

    systematic_risk = pd.DataFrame(data=[sys_risk], 
                               columns=[portfolio.name + ' Systematic Risk'])
    
    return systematic_risk


# ------------------------- IDIOSYNCRATIC RISK ------------------------- #

def idiosyncratic_risk(portfolio, Index, start_date=None, end_date=None):
    """Estimate of the Portfolio's idiosyncratic (unsystematic/
    undiversifiable) risk. Risk is expressed as a percentage.
    
        IDIOSYNCRATIC RISK = SQRT( TOTAL VAR - SYSTEMATIC VAR ) * 100
            s.t. systematic var = ( systematic risk )^2
    
    Parameters
    ----------
    portfolio : Portfolio object.
        A portfolio for correlation analysis.
    Index : Stock object.
        A stock representing the provided composite market index.
    start_date : String.
        Sets the start of the risk study. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the risk study. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """
    
    port_ret = daily_portfolio_combined_rate_of_return(portfolio, start_date, end_date)
    total_var = port_ret.var().values[0]

    sys_risk = systematic_risk(portfolio, Index, start_date, end_date).values[0][0]
    sys_var = sys_risk ** 2

    idio_risk = np.sqrt( total_var - sys_var ) * 100

    idiosyncratic_risk = pd.DataFrame(data=[idio_risk], 
                               columns=[portfolio.name + ' Idiosyncratic Risk'])
    
    return idiosyncratic_risk


# ------------------------- SHARPE RATIO ------------------------- #

def sharpe_ratio(portfolio, risk_free_return, start_date=None, end_date=None):
    """Calculates the Sharpe Ratio, indicating the degree to which the
    portfolio provides excess returns above and beyond the excess risks.
    
        IDIOSYNCRATIC RISK = SQRT( TOTAL VAR - SYSTEMATIC VAR ) * 100
            s.t. systematic var = ( systematic risk )^2
    
    Parameters
    ----------
    portfolio : Portfolio object.
        A portfolio for correlation analysis.
    risk_free_return : Float.
        The annual risk free rate of return.
    start_date : String.
        Sets the start of the risk study. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the risk study. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A Pandas DataFrame object.
    """
    
    annual_risk_free_return = 0.79
    annual_return = average_annual_portfolio_rate_of_return(portfolio, start_date, end_date).values[0][0]
    returns_std = portfolio_annual_standard_deviation(portfolio, start_date, end_date).values[0][0]

    sharpe_ratio = ( annual_return - annual_risk_free_return) / returns_std

    sharpe_ratio = pd.DataFrame(data=[sharpe_ratio], columns=[portfolio.name + ' Sharpe Ratio'])
    
    return sharpe_ratio


# ------------------------- MARKOWITZ EFFICIENT FRONTIER ------------------------- #

def markowitz_efficient_frontier(portfolio, density, risk_free_return, figure_path, start_date=None, end_date=None):
    """Generates the efficient frontier of the portfolio and marks the
    existing composition against the alternatives. All data generated
    in the analysis is returned as a dataframe.
    
    Parameters
    ----------
    portfolio : Portfolio object.
        A portfolio for correlation analysis.
    density : Integer.
        The density of the simulated frontier.
    risk_free_return : Float.
        The annual risk free rate of return.
    start_date : String.
        Sets the start of the risk study. Format 'YYYY-MM-DD'.
    end_date : String.
        Sets the end of the risk study. Format 'YYYY-MM-DD'.
    
    Returns
    -------
    A List of Dictionaries.
    """
    
    returns = list()
    risks = list()
    sharpe = list()
    data = list()
    
    for i in range(density):
        rand = random(len(portfolio.stocks))
        weights = np.reshape(rand / rand.sum(), (len(portfolio.stocks),1))
        
        Simulation = PortfolioModel(portfolio.name, portfolio.stocks, weights)
        
        returns.append(average_annual_portfolio_rate_of_return(Simulation, start_date, end_date).values[0][0])
        risks.append(portfolio_annual_standard_deviation(Simulation, start_date, end_date).values[0][0])
        sharpe.append(sharpe_ratio(Simulation, risk_free_return, start_date, end_date).values[0][0])
        data.append(dict({"Weights": weights, "Return": returns[i], "Risk": risks[i], "Sharpe": sharpe[i]}))
    
    fig = plt.figure(figsize=[6, 6])
    
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    
    plt.scatter(risks, returns, c=sharpe, cmap=plt.cm.winter)
    plt.title(portfolio.name + ' Efficient Frontier')
    plt.xlabel('Annualized Standard Deviation')
    plt.ylabel('Annualized Return')
    
    norm = plt.Normalize(np.min(sharpe), np.max(sharpe))
    mappable =  ScalarMappable(norm=norm, cmap=plt.cm.winter)
    mappable.set_array([])
    cbar = plt.colorbar(mappable)
    cbar.ax.set_title("")
    cbar.set_label('Sharpe Ratio')
    
    portfolio_risk = portfolio_annual_standard_deviation(portfolio, start_date, end_date).values[0][0]
    portfolio_return = average_annual_portfolio_rate_of_return(portfolio, start_date, end_date).values[0][0]
    plot = plt.scatter(portfolio_risk, portfolio_return, c='red')
    
    plt.savefig(figure_path + '.svg', transparent=False)
    plt.savefig(figure_path + '.png', transparent=False)
    plt.close()
    
    return data


