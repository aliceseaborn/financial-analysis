#!/usr/bin/env python
"""Object representing a single tradable equity.
"""

...

__author__ = "Austin Dial, Alice Seaborn"

__version__ = "0.0.0"
__maintainer__ = "Alice Seaborn"
__email__ = "adial@mail.bradley.edu"
__status__ = "Prototype"



class StockModel(object):
    
    def __init__(self, ticker, name, history):
        self.ticker = ticker
        self.name = name
        self.history = history
