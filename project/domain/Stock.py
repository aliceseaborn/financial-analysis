#!/usr/bin/env python
"""Object representing a single tradable equity.
"""

from datetime import datetime as dt
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np

__author__ = "Alice Seaborn"

__version__ = "0.0.0"
__maintainer__ = "Alice Seaborn"
__email__ = "seaborn.dev@gmail.com"
__status__ = "Prototype"


class Stock(object):
    def __init__(self, ticker, name, history):
        try:
            self.ticker = str(ticker)
            self.name = str(name)
            if isinstance(history, pd.DataFrame):
                self.history = history
            else:
                raise TypeError(
                    f"History must be a Pandas DataFrame. Passed {type(history)}."
                )
        except TypeError:
            raise TypeError(
                f"Types must be (str, str, pd.DataFrame). Improper types"
                "({type(ticker)}, {type(name)}, {type(history)}) passed."
            )

    def annualization_factor(self, start_date=None, end_date=None):
        """Calculates the average number of trading days within the given date range
        for the stock history. If the date range is less than one year, then a default
        annualization of 250 is returned.
        """

        if not start_date:
            start_date = self.history.iloc[0, 0]
        if not end_date:
            end_date = self.history.iloc[len(self.history) - 1, 0]

        start_dt = dt.strptime(start_date, "%Y-%m-%d")
        end_dt = dt.strptime(end_date, "%Y-%m-%d")

        time_delta = relativedelta(end_dt, start_dt)
        years = time_delta.years + (time_delta.months / 12) + (time_delta.days / 365)

        selection = self.history.query(
            f"Date >= '{start_date}' and Date " f"<= '{end_date}'"
        ).reset_index(drop=True)

        sample_length = len(selection)

        if years < 1:
            return 250
        else:
            return sample_length / years

    def daily_rate_of_return(self, mode="simple", start_date=None, end_date=None):
        """Series of daily rates of return in either simple ('s') or logarithmic ('l')
        mode between the given dates. Date format is "yyyy-mm-dd". Returns are not
        expressed as a percentage.

            DAILY LOG ROR = LOG( CLOSE / CLOSE.SHIFT(1) )
            DAILY SIMPLE ROR = ( CLOSE / CLOSE.SHIFT(1) ) - 1
                s.t. ROR - rate of return
        """

        if not start_date:
            start_date = self.history.iloc[0, 0]
        if not end_date:
            end_date = self.history.iloc[len(self.history) - 1, 0]

        selection = self.history.query(
            f"Date >= '{start_date}' and Date " f"<= '{end_date}'"
        ).reset_index(drop=True)
        dates = selection.iloc[:, 0]

        if mode == "simple" or mode == "s":
            returns = (selection.iloc[:, 5] / selection.iloc[:, 5].shift(1)) - 1
        elif mode == "logarithmic" or mode == "l":
            returns = np.log(selection.iloc[:, 5] / selection.iloc[:, 5].shift(1))
        else:
            raise ValueError(f"Mode: {mode} is unacceptable.")

        daily_ror = pd.concat([dates, returns], axis=1)
        daily_ror = daily_ror.rename(
            columns={"Date": "Date", "Adj Close": self.ticker + " ROR"}
        )

        return daily_ror

    def average_daily_rate_of_return(
        self, mode="simple", start_date=None, end_date=None
    ):
        """Average daily simple or logarithmic rate of return between the given
        dates. Average return is not expressed as a percentage. Date format is
        "yyyy-mm-dd".

            AVERAGE DAILY ROR = MEAN( DAILY ROR ) * 100
                s.t. ROR - rate of return
        """

        daily_ror = self.daily_rate_of_return(mode, start_date, end_date)
        average_daily_ror = daily_ror.iloc[:, 1].mean()

        return average_daily_ror

    def annual_rates_of_return(self, years):
        """Year by year simple rates of returns for the years specified. Rates of
        return are expressed as a percentage. Format for year is ["1900",  "1901",
        ... "YYYY"].

            FOR YEAR IN YEARS:
                RATE OF RETURN = (ENDING PRICE / STARTING PRICE) * 100
        """

        year_range = list()
        annual_returns = list()

        for i in range(len(years) - 1):
            selection = self.history.query(
                f"Date >= '{years[i]}' and Date <= '{years[i+1]}'"
            )
            annual_ror = round(
                ((selection.iloc[len(selection) - 1, 5] / selection.iloc[0, 5]) - 1)
                * 100,
                2,
            )

            year_range.append(f"{years[i]} to {years[i+1]}")
            annual_returns.append(annual_ror)

        return pd.DataFrame({"Year": year_range, "Rate of Return": annual_returns})

    def average_annual_rate_of_return(self, years):
        """Average annual simple rate of return between the given
        dates. Date format is "yyyy-mm-dd".

            AVERAGE ANNUAL ROR = AVE( ANNUAL RATES OF RETURN )
                s.t. ROR - rate of return
        """

        returns = self.annual_rates_of_return(years).iloc[:, 1].values
        average_returns = np.average(returns)

        return average_returns

    def annual_volatility(self, start_date=None, end_date=None):
        """Annual standard deviation of the stock returns over the specified period.
        The standard deviation is expressed as a percentage. Date format is
        "yyyy-mm-dd".

            ST. DEV = STD( DAILY STOCK ROR ) * AF
                s.t. ROR - rate of return
                     AF - Annualization Factor
        """

        annualization = self.annualization_factor(start_date, end_date)

        stock_ror = self.daily_rate_of_return("simple", start_date, end_date)
        stock_std = stock_ror.iloc[:, 1].std() * annualization

        return stock_std

    def annual_variance(self, start_date=None, end_date=None):
        """Annual variance of the stock returns over the specified period. The
        variance is not expressed as a percentage. Date format is "yyyy-mm-dd".

            VARIANCE = VAR( DAILY STOCK ROR ) * AF
                s.t. ROR - rate of return
                     AF - Annualization Factor
        """

        annualization = self.annualization_factor(start_date, end_date)

        stock_ror = self.daily_rate_of_return("simple", start_date, end_date)
        stock_var = stock_ror.iloc[:, 1].var() * annualization

        return stock_var

    def beta(self, index, start_date=None, end_date=None):
        """The beta for the stock against the market index, provided as
        a stock object. Date format is "yyyy-mm-dd".

            BETA = COV( STOCK RETURNS, INDEX RETURNS ) / VAR( INDEX )
        """

        stock_ret = self.daily_rate_of_return("logarithmic", start_date, end_date)
        index_ret = index.daily_rate_of_return("logarithmic", start_date, end_date)

        covariance = np.cov(stock_ret.iloc[:, 1][1:], index_ret.iloc[:, 1][1:])[0, 0]
        variance = np.var(index_ret.iloc[:, 1])

        beta = covariance / variance

        return beta
