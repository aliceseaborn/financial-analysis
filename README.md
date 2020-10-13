# Financial Analysis Tools

<!-- [![CircleCI](https://circleci.com/gh/Adial314/udemy-testing-in-python.svg?style=svg)](https://circleci.com/gh/Adial314/udemy-testing-in-python) -->

A library of custom financial analysis tools that were developed during a finance courses on financial analysis. This library includes functions for analyzing individual stocks as well as portfolios of equities. At this time, options and bonds are not included in this analysis package. As I develop more financial code, I will update this library by extending and expanding the objects therein. 

This repository contains:

1. The repository requirements in `requirements.txt`
2. Full end-to-end and unit tests in `tests/`
3. CircleCI configurations and Makefiles for running tests against the library
4. Library of financial objects and tools in `project/`



## Table of Contents

- [Background](#background)
- [Usage](#usage)
    - [Analyze Stock](#analyze-stock)
    - [Analyze Portfolio](#analyze-portfolio)



## Background

This library focuses on providing financial analysis tools. The technical analysis of equities in Python is a difficult and quite often messy undertaking. The objects created herein are designed to do the heavy lifting for the user without sacrificing configurability. The user can simply direct the analysis and consume the output without having to worry about setting up data pipelines into various stock data providers. This library was built on top of Pandas for the analysis of DataFrames instead of using POPOs. Consequently, some operations might be compute intensive but the code has been optimized for Pandas wherever possible.

There are two types of domain objects that can be analyzed by this library: Stock and Portfolio objects. To avoid confusion with instances of the Stock object, the domain model representation is named StockModel.

A Stock object has a descriptive string name of the equity as well as a string ticker to identify it. In addition to descriptive information, each stock contains data of its price history in the form of a Pandas DataFrame object.

A Portfolio object also has a descriptive string name to identify it. Each Portfolio contains a list of Stock objects representing its investments as well as a Numpy array of the proportion of the Portfolio's funds that are dedicated to each Stock.



## Usage

Install the requirements in a python environment using the included requirements file. Then navigate into the `project` directory.

```sh
pip install -r requirements.txt
```

Import the `StockModel` and the `PortfolioModel` model from the `domain` folder and instantiate each. In this example, a Portfolio composed of Proctor & Gamble Co. (PG) and Microsoft (MSFT) is created with `25%` investment in PG and `75%` investment in MSFT.

```python
>>> from domain.Stock import StockModel
>>> from domain.Portfolio import PortfolioModel
>>> PG = StockModel("PG", "Proctor & Gamble Co.", pg_history)
>>> MSFT = StockModel("MSFT", "Microsoft Inc.", msft_history)
>>> BlueChips = PortfolioModel("Blue Chips", [PG, MSFT], np.array([ [0.25], [0.75] ]))
```

### Analyze Stock

In order to perform an analysis method against a Stock object, first import the method from the `stock_analysis` module and then pass as function arguments the Stock as well as the analysis settings. Below, for example, we will analyze the annual variance of a Stock object within the last ten years starting on January first.

```python
>>> from stock_analysis import stock_annual_variance
>>> stock_annual_variance(PG, "01/01/2010", "01/01/2020")
	PG Variance
0	0.046239
```

### Analyze Portfolio

Analyzing a Portfolio object follows the same logic as the analysis of a Stock object, simply import the desired tool from the `portfolio_analysis` module and pass the Portfolio to the tool as an argument. In this example, we will obtain the average annual rate of return for the *BlueChips* Portfolio created above.

```python
>>> average_annual_portfolio_rate_of_return(BlueChips, "01/01/2010", "01/01/2020")
Blue Chips Average Annual Return
0	12.895175
```
