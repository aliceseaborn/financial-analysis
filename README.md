# Financial Analysis Tools

<!-- [![CircleCI](https://circleci.com/gh/Adial314/udemy-testing-in-python.svg?style=svg)](https://circleci.com/gh/Adial314/udemy-testing-in-python) -->

A library of custom financial analysis tools that were originally developed during a Udemy course on financial analysis with Python. This library includes objects for representing individual stocks and portfolios of stocks. At this time, options and bonds are not included in this analysis package. As I develop more financial code, I will update this library by extending and expanding the objects therein and the functionality they offer.

This repository contains:

1. The repository requirements in `requirements.txt`
2. Library of financial objects and tools in `project/`
3. Full end-to-end and unit tests in `project/tests/`
4. CircleCI configurations and Makefiles for running tests against the library


## Table of Contents

- [Background](#background)
- [Usage](#usage)
    - [Analyze Stock](#analyze-stock)
    - [Analyze Portfolio](#analyze-portfolio)
- [Testing](#testing)
- [Cleansing](#cleansing)


## Background

This library focuses on providing financial analysis tools. The technical analysis of equities in Python is a difficult and quite often messy undertaking. The objects created herein are designed to do the heavy lifting for the user without sacrificing configurability. The user can simply direct the analysis and consume the output without having to worry about setting up data pipelines into various stock data providers. This library was built on top of Pandas for the analysis of DataFrames instead of using POPOs. Consequently, some operations might be compute intensive but the code has been optimized for Pandas wherever possible.

A Stock object has a descriptive string name of the equity as well as a string ticker to identify it. In addition to descriptive information, each stock contains data of its price history in the form of a Pandas DataFrame object. This data must be supplied as a CSV that the object can consume.

A Portfolio object also has a descriptive string name to identify it. Each Portfolio contains a list of Stock objects representing its investments as well as a Numpy array of the proportion of the Portfolio's funds that are dedicated to each Stock. Portfolios offer users the option to provided analyses focused on the individual stocks or the portfolio itself.


## Usage

Install the requirements in a python environment using the included requirements file. Then navigate into the `project/` directory.

```BASH
pip3 install -r requirements.txt
```

### Analyze Stock

Instantiate a stock object by providing a stock ticker, company name, and a path to a CSV file containing the stock's price history. Once the stock object has been instantiated may call its member functions to perform the desired analysis as shown below.

```PYTHON
import numpy as np
import pandas as pd

from project.domain.Stock import Stock

history = pd.read_csv("data/MSFT.csv")
MSFT = Stock("MSFT", "Microsoft Incorporated", history)

start_date = "1900-01-01"
end_date = "1901-01-01"

MSFT.annual_volatility(start_date, end_date)
```

### Analyze Portfolio

Similarly, member functions for portfolios provide analysis tools as member function that act against the object itself.

```PYTHON
import numpy as np
import pandas as pd

from project.domain.Stock import Stock
from project.domain.Portfolio import Portfolio

stock_path = "data/{}.csv"
weights = np.array([[0.40], [0.60]])
tickers = ["PG", "MSFT"]
companies = ["Proctor & Gamble Company", "Microsoft Incorporated"]
entries = dict(zip(tickers, companies))
stocks = list()
for ticker, company in entries.items():
    	history = pd.read_csv(stock_path.format(ticker))
    	stocks.append(Stock(ticker, company, history))
BlueHaven = Portfolio("BlueHaven", stocks, weights)

start_date = "1900-01-01"
end_date = "1901-01-01"

BlueHaven.covariance_matrix(start_date, end_date)
...
```


## Testing

This project uses the `pytest` module/framework for automated testing of project code. From within the root folder (`/`, the parent of `project/`), run any of the following commands to interface with the pytest structure.

```BASH
pytest --collect-only 									# Collect but do not execute all tests
pytest                									# Run all tests with standard verbosity
pytest -v             									# ^+ increased verbosity
pytest --cov=project  									# ^+ include coverage tests
pytest --pep8 											# ^+ test code for PEP8 compliance
pytest --html=path/to/report.html --self-contained-html # ^+ create HTML testing report
```


## Cleaning

In addition to the automated testing tools described [above](#Testing), this project also utilizes Python Black for automatied cleansing of Python code for PEP8 standards. Black will automatically format the python code to make it more readable. Pylint will then check for more advanced errors. Lastly, pytest will verify compliance with PEP8 standards.

```BASH
black project/hello.py
pylint --disable=R,C project/hello.py
pytest --pep8 --disable-warnings project/hello.py
```



*Alice Seaborn. May 1. 2021.*
