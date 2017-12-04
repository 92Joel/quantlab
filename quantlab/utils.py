from IPython.display import display
import pandas_datareader.data as web
import datetime as dt
import pandas as pd

def print_table(table, name=None, fmt=None):


    """
    Pretty print a pandas DataFrame.

    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.

    Parameters
    ----------
    table : pandas.Series or pandas.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    fmt : str, optional
        Formatter to use for displaying table elements.
        E.g. '{0:.2f}%' for displaying 100 as '100.00%'.
        Restores original setting after displaying.
    """

    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if fmt is not None:
        prev_option = pd.get_option('display.float_format')
        pd.set_option('display.float_format', lambda x: fmt.format(x))

    if name is not None:
        table.columns.name = name
    pd.set_option('display.max_rows', len(table))
    display(table)
    pd.reset_option('display.max_rows')

    if fmt is not None:
        pd.set_option('display.float_format', prev_option)

def get_default_bench(start, end, ticker = 'SPY'):
    start = pd.to_datetime(start) - pd.offsets.BDay(1)
    bench = web.DataReader(ticker, 'google', start, end)
    bench_rets = bench.Close.pct_change().iloc[1:]
    return bench_rets

def percentage(x, pos):
    """
    Adds percentage sign to plot ticks.
    """
    return '%.0f%%' % x

def two_dec_places(x, pos):
    """
    Adds 1/100th decimal to plot ticks.
    """

    return '%.2f' % x
