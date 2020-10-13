#!/usr/bin/env python
"""Object representing a portfolio of tradable equities.
"""

...

__author__ = "Austin Dial, Alice Seaborn"

__version__ = "0.0.0"
__maintainer__ = "Alice Seaborn"
__email__ = "adial@mail.bradley.edu"
__status__ = "Prototype"



class PortfolioModel(object):
    
    def __init__(self, name, stocks, weights):
        self.name = name
        self.stocks = stocks
        self.weights = weights
