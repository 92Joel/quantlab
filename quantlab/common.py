import pandas as pd
import numpy as np
from tia.bbg import LocalTerminal as bbg
import tia.bbg.datamgr as dm
import matplotlib.pyplot as plt
import tia
from datetime import date
import scipy

def get_members(indx):
    '''
    Return bloomberg index nembers
    '''
    stocks = bbg.get_reference_data(indx,'INDX_MEMBERS')
    stocks=stocks.as_frame()
    ukx=stocks['INDX_MEMBERS'][0]
    stocks = map(lambda x: x+" Equity", list(ukx["Member Ticker and Exchange Code"]))
    return stocks

def get_OHLCV(universe, start, end = '', as_frame = False):
    '''
    Return OHLCV data for a given universe
    ---------------------------------------
    start must be a date string in 'YYYY-MM-DD' format

    end defaults to the current day
    
    If as_frame = True returns a multi-index dataframe. Otherwise, a panel is returned.      
    '''
    df = pd.DataFrame()
    mgr     = dm.BbgDataManager()
    fields = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME']
    for field in fields:
        tmp = mgr[universe].get_historical(field, start, end, period)
        tmp.columns = pd.MultiIndex.from_product([field, universe], names=['Field', 'Stock'])
        df=pd.concat([df,tmp],axis=1)
        data = {}
    for field in df.columns.levels[0]:
        data[field] = df[field]
    pan = pd.Panel(data)
    pan = pan.transpose(2,1,0)
    if as_frame:
        return pan.to_frame()
    else:
        return pan

