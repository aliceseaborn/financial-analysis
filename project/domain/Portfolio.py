#!/usr/bin/env python
"""Object representing a portfolio of tradable equities.
"""

import pandas as pd

import numpy as np
from numpy.random import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable

__author__ = "Austin Dial, Alice Seaborn"

__version__ = "0.0.0"
__maintainer__ = "Alice Seaborn"
__email__ = "seaborn.archipelago@gmail.com"
__status__ = "Prototype"



class Portfolio(object):
    
    def __init__(self, name, stocks, weights):
        self.name = name
        self.stocks = stocks
        self.weights = weights
    
    
    def daily_individual_rates_of_return(self, start_date=None, end_date=None):
        """Calculates the daily return histories for each individual stock
        in the portfolio.

            FOR STOCK IN PORTFOLIO:
                STOCK ROR = LOG( CLOSE / CLOSE.SHIFT(1) )
        """
    
        dataframes = list([])
        for stock in self.stocks:
            returns = stock.daily_rate_of_return('l', start_date, end_date)
            df = pd.DataFrame(data=returns.iloc[:,1].to_numpy(), 
                              columns=[stock.ticker + ' Returns'])
            dataframes.append(df)

        dataframes.insert(0, pd.DataFrame(data=returns['Date'], 
                                          columns=['Date']))

        return pd.concat(dataframes, axis=1)


    def daily_combined_rate_of_return(self, start_date=None, end_date=None):
        """Calculates the daily return history for portfolio between the
        given dates.
        """

        # Calculate the daily rates of return for each stock
        indiv_ret = self.daily_individual_rates_of_return(start_date, end_date)

        # Distribute weights to returns
        weighted_indiv_ret = pd.DataFrame()
        for i in range(len(indiv_ret.columns)-1):
            weighted_indiv_ret[indiv_ret.columns[i+1]] = indiv_ret[indiv_ret.columns[i+1]].astype(float) * self.weights[i][0]

        # Calculate the portfolio's daily rate of return
        total_returns = pd.Series(data=weighted_indiv_ret.sum(axis=1), 
                                  name=self.name + " Returns")
        dates = pd.Series(data=indiv_ret['Date'], name="Date")
        port_ret = pd.concat([dates, total_returns], axis=1)

        return port_ret
    
    
    def daily_excess_rates_of_return(self, risk_free_rate, start_date=None, end_date=None):
        """Calculates the combined portfolio returns after considering the 
        risk free rate of return. This represents the excess returns the
        portfolio yields as a result of taking risk.

            EXCESS PORTFOLIO RETURNS = DAILY PORTFOLIO COMBINED
                RATE OF RETURNS - ( RISK FREE RATE OF RETURNS / 250 )
        """

        excess_daily_returns = pd.DataFrame()
        daily_returns = self.daily_combined_rate_of_return(start_date, end_date)
        excess_daily_returns['Date'] = daily_returns.iloc[:,0]
        excess_daily_returns[self.name + ' Excess Daily Returns'] = daily_returns.iloc[:,1] - ( risk_free_rate / 250 )

        return excess_daily_returns


    def average_daily_rate_of_return(self, start_date=None, end_date=None):
        """Average daily rate of return for the given portfolio between the 
        given dates. Average return is expressed as a percentage.

            AVE. DAILY PORT ROR = MEAN( PORT ROR ) * 100
        """

        portfolio_ror = self.daily_combined_rate_of_return(start_date, end_date)
        average_daily_portfolio_ror = portfolio_ror[self.name + ' Returns'].mean() * 100

        return average_daily_portfolio_ror


    def average_annual_rate_of_return(self, start_date=None, end_date=None):
        """Average annual rate of return for the given portfolio between the 
        given dates.
        """

        ave_annual_return = self.average_daily_rate_of_return(start_date, end_date) * 250

        return ave_annual_return


    def individual_variances(self, start_date=None, end_date=None):
        """Variances for each stock in the portfolio.
        """

        data = dict()
        for stock in self.stocks:
            data[stock.ticker + ' Variance'] = [stock.annual_variance(start_date, end_date)]

        individual_vars = pd.DataFrame(data)

        return individual_vars
    
    
    def covariance_matrix(self, start_date=None, end_date=None):
        """Covariance matrix of the provided stocks.

        The covariance is calculated by calling .cov() against the individual
        daily rates of return for the portfolio's stocks.
        """

        indiv_ret = self.daily_individual_rates_of_return(start_date, end_date)
        cov_matrix = indiv_ret.cov()

        return cov_matrix


    def variance(self, start_date=None, end_date=None):
        """The annual variance of the portfolio.

            VARIANCE = W^T * COVARIANCE * W
                s.t. W = Weights
        """

        cov_matrix = self.covariance_matrix(start_date, end_date)

        portfolio_variance = np.matmul( np.matmul( self.weights.T, cov_matrix.to_numpy() ), 
                             self.weights )[0,0] * 100

        return portfolio_variance


    def annual_standard_deviation(self, start_date=None, end_date=None):
        """The annual standard deviation of the portfolio returns.

            STANDARD DEVIATION = STD( PORTFOLIO DAILY RETURNS ) * 250
        """

        returns = self.daily_combined_rate_of_return(start_date, end_date)
        returns_std = returns.iloc[:,1].std() * 250

        return returns_std


    def volatility(self, start_date=None, end_date=None):
        """The annual volatility of the portfolio expressed as a percentage.

            VOLATILITY = SQRT( W^T * COVARIANCE * W )
                s.t. W = Weights
        """

        variance = self.variance(start_date, end_date)
        port_volatility = np.sqrt( variance ) * 100

        return port_volatility


    def correlation_matrix(self, start_date=None, end_date=None):
        """Correlation matrix of the provided stocks.

        The correlation is calculated by calling .corr() against the individual
        daily rates of return for the portfolio's stocks.
        """

        indiv_ret = self.daily_individual_rates_of_return(start_date, end_date)
        corr_matrix = indiv_ret.corr()

        return corr_matrix


    def individual_betas(self, Index, start_date=None, end_date=None):
        """The betas for each stock in the Portfolio against the provided market index.

            BETA = COV( STOCK RETURNS, INDEX RETURNS ) / VAR( INDEX RETURNS )
        """
        
        data = dict()
        for stock in self.stocks:
            data[stock.name] = stock.beta(Index, start_date, end_date)

        return pd.DataFrame(data=data, index=[0])


    def beta(self, Index, start_date=None, end_date=None):
        """The beta for the Portfolio against the provided market index.

            BETA = COV( PORTFOLIO RETURNS, INDEX RETURNS ) / VAR( INDEX )
        """

        portfolio_ret = self.daily_combined_rate_of_return(start_date, end_date)
        index_ret = Index.daily_rate_of_return('l', start_date, end_date)

        covariance = np.cov(portfolio_ret.iloc[:,1], index_ret.iloc[:,1])[0,0]
        variance = np.var(index_ret.iloc[:,1])

        portfolio_beta = covariance / variance

        return portfolio_beta


    def systematic_risk(self, Index, start_date=None, end_date=None):
        """Estimate of the Portfolio's systematic (undiversifiable) risk.
        Risk is expressed as a percentage.

            SYSTEMATIC RISK = PORTFOLIO BETA * STD( INDEX ) * 100
        """

        beta = self.beta(Index, start_date, end_date)

        index_ret = Index.daily_rate_of_return('l', start_date, end_date)
        index_std = index_ret.std().values[0]
        
        systematic_risk = beta * index_std * 100

        return systematic_risk


    def idiosyncratic_risk(self, Index, start_date=None, end_date=None):
        """Estimate of the Portfolio's idiosyncratic (unsystematic/
        undiversifiable) risk. Risk is expressed as a percentage.

            IDIOSYNCRATIC RISK = SQRT( TOTAL VAR - SYSTEMATIC VAR ) * 100
                s.t. systematic var = ( systematic risk )^2
        """

        port_ret = self.daily_combined_rate_of_return(start_date, end_date)
        total_var = port_ret.var().values[0]

        sys_risk = self.systematic_risk(Index, start_date, end_date)
        sys_var = sys_risk ** 2

        idiosyncratic_risk = np.sqrt(np.abs(total_var - sys_var))

        return idiosyncratic_risk


    def sharpe_ratio(self, risk_free_return, start_date=None, end_date=None):
        """Calculates the Sharpe Ratio, indicating the degree to which the
        portfolio provides excess returns above and beyond the excess risks.

            IDIOSYNCRATIC RISK = SQRT( TOTAL VAR - SYSTEMATIC VAR ) * 100
                s.t. systematic var = ( systematic risk )^2
        """
        
        annual_return = self.average_annual_rate_of_return(start_date, end_date)
        returns_std = self.annual_standard_deviation(start_date, end_date)

        sharpe_ratio = ( annual_return - risk_free_return) / returns_std

        return sharpe_ratio


    def markowitz_efficient_frontier(self, density, risk_free_return, figure_path, start_date=None, end_date=None):
        """Generates the efficient frontier of the portfolio and marks the
        existing composition against the alternatives. All data generated
        in the analysis is returned as a dataframe.
        """

        returns = list()
        risks = list()
        sharpe = list()
        data = list()

        for i in range(density):
            rand = random(len(self.stocks))
            weights = np.reshape(rand / rand.sum(), (len(self.stocks),1))

            Simulation = Portfolio(self.name, self.stocks, weights)

            returns.append(Simulation.average_annual_rate_of_return(start_date, end_date))
            risks.append(Simulation.annual_standard_deviation(start_date, end_date))
            sharpe.append(Simulation.sharpe_ratio(risk_free_return, start_date, end_date))
            data.append(dict({"Weights": weights, "Return": returns[i], "Risk": risks[i], "Sharpe": sharpe[i]}))

        fig = plt.figure(figsize=[6, 6])

        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1.0)

        plt.scatter(risks, returns, c=sharpe, cmap=plt.cm.winter)
        plt.title(self.name + ' Efficient Frontier')
        plt.xlabel('Annualized Standard Deviation')
        plt.ylabel('Annualized Return')

        norm = plt.Normalize(np.min(sharpe), np.max(sharpe))
        mappable =  ScalarMappable(norm=norm, cmap=plt.cm.winter)
        mappable.set_array([])
        cbar = plt.colorbar(mappable)
        cbar.ax.set_title("")
        cbar.set_label('Sharpe Ratio')

        portfolio_risk = self.annual_standard_deviation(start_date, end_date)
        portfolio_return = self.average_annual_rate_of_return(start_date, end_date)
        plot = plt.scatter(portfolio_risk, portfolio_return, c='red')

        plt.savefig(figure_path + '.svg', transparent=False)
        plt.savefig(figure_path + '.png', transparent=False)
        plt.close()

        return data





