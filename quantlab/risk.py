import pandas as pd 
import numpy as np 
import math

from backtest import Backtest

class Risk(Backtest):
    def __init__(self, prices, name, holdings = None, risk_mgmt_active = False,
        risk_mgmt_periods = None, risk_mgmt_inequality = None, risk_mgmt_method = None):

        Backtest.__init__(self, prices, name, holdings = None, risk_mgmt_active = False,
        risk_mgmt_periods = None, risk_mgmt_inequality = None, risk_mgmt_method = None):

        