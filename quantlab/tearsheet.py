import pandas_datareader.data as web
import pandas as pd

from .statistics import Statistics
from . import plotting
from .plotting import plotting_context
from . import utils
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib
import seaborn as sns
from pandas import ExcelWriter

from periods import ANNUALIZATION_FACTORS, APPROX_BDAYS_PER_YEAR
from periods import DAILY, WEEKLY, MONTHLY, ANNUAL


class TearSheet(Statistics):
    def __init__(self, returns, period='daily', benchmark=None, default_bench=True, risk_free=5,
                 title=None, rolling_stats=False):
        super(TearSheet, self).__init__(
            returns, period, benchmark, default_bench, risk_free)
        self.title = title
        self.rolling_stats = rolling_stats

    def results_table(self, display = True):
        table = pd.Series(index=BASIC_STATS)
        for stat in BASIC_STATS:
            if stat == 'cumulative_returns':
                table[stat] = getattr(self, stat)().values[-1]
            else:
                table[stat] = getattr(self, stat)()
        table.name = 'Results'

        if display:
            utils.print_table(table)
        return table

    @plotting_context
    def plot_returns(self, ax=None, display=True, **kwargs):

        if ax is None:
            ax = plt.gca()
        ax.set_label('')
        ax.set_ylabel('Return')
        ax.set_title('Returns', size = 14)

        self.returns.plot(ax=ax, color='forestgreen', **kwargs)

        if display:
            plt.show()
        return ax

    @plotting_context
    def plot_cumulative_returns(self, ax=None, title=None, volatility_match=False, log_returns=False, display=True, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.set_label('')
        ax.set_xlabel('')
        ax.set_ylabel('Return')

        if title is not None:
            ax.set_title(title, size = 14)
        else:
            ax.set_title('Cumulative Returns', size = 14)
        ax.set_yscale('log' if log_returns else 'linear')
        
        y_axis_formatter = FuncFormatter(utils.two_dec_places)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        returns = self.returns.copy()

        if volatility_match and self.benchmark is None:
            raise ValueError(
                'volatility_match requires benchmark returns to be instantiated in the object.')

        elif volatility_match and self.benchmark is not None:
            bench_vol = self.benchmark.loc[returns.index].std()
            returns = (returns / returns.std()) * bench_vol

        cumulative_returns = self.cumulative_returns(returns, starting_val=1)
        cumulative_returns.plot(ax=ax, color='gray', **kwargs)

        if self.benchmark is not None:
            bench_returns = self.cumulative_returns(
                self.benchmark, starting_val=1)
            bench_returns.plot(ax=ax, color='g', **kwargs)

        ax.legend(['Strategy Returns', 'Benchmark Returns'],
                  loc='best')

        ax.axhline(1.0, linestyle='--', color='black', lw=2)
        if display:
            plt.show()
        return ax

    @plotting_context
    def plot_rolling_sharpe(self, ax=None, display=True, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.set_label('')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Rolling Sharpe Ratio (6 Month)')

        rolling_sharpe = self.returns.rolling(
            126).apply(lambda x: self.sharpe(x))
        rolling_sharpe.plot(ax=ax, color='orangered', **kwargs)

        ax.axhline(
            rolling_sharpe.mean(),
            color='steelblue',
            linestyle='--',
            lw=3)

        y_axis_formatter = FuncFormatter(utils.two_dec_places)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        ax.legend(['Sharpe', 'Average'],
                  loc='best')

        if display:
            plt.show()
        return ax

    @plotting_context
    def plot_rolling_sortino(self, ax=None, display=True, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.set_label('')
        ax.set_ylabel('Sortino Ratio')
        ax.set_title('Rolling Sortino Ratio (6 Month)')

        rolling_sortino = self.returns.rolling(
            126).apply(lambda x: self.sortino(x))
        rolling_sortino.plot(ax=ax, color='orangered', **kwargs)

        ax.axhline(
            rolling_sortino.mean(),
            color='steelblue',
            linestyle='--',
            lw=3)

        y_axis_formatter = FuncFormatter(utils.two_dec_places)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        ax.legend(['Sortino', 'Average'],
                  loc='best')

        if display:
            plt.show()
        return ax

    @plotting_context
    def plot_monthly_returns_heatmap(self, ax=None, display=True, **kwargs):
        if ax is None:
            ax = plt.gca()

        returns_table = self.aggregate_returns(MONTHLY)
        returns_table = returns_table.unstack().round(3)

        sns.heatmap(
            returns_table.fillna(0) *
            100.0,
            annot=True,
            annot_kws={
                "size": 9},
            alpha=1.0,
            center=0.0,
            cbar=False,
            cmap=matplotlib.cm.RdYlGn,
            ax=ax, **kwargs)

        ax.set_ylabel('Year')
        ax.set_xlabel('Month')
        ax.set_title("Monthly returns (%)", size = 14)

        if display:
            plt.show()

        return ax

    @plotting_context
    def plot_rolling_beta(self, ax=None, display=True, **kwargs):
        if ax is None:
            ax = plt.gca()

        rolling_beta12M = self.returns.rolling(
            252).apply(lambda x: self.beta(x))
        rolling_beta6M = self.returns.rolling(
            126).apply(lambda x: self.beta(x))

        ax.set_label('')
        ax.set_ylabel('Beta')
        ax.set_title('Rolling Beta to Benchmark', size = 14)

        rolling_beta12M.plot(ax=ax, color='steelblue')
        rolling_beta6M.plot(ax=ax, color='gray')

        ax.legend(['12 Month', '6 Month'],
                  legend='best')

        if display:
            plt.show()
        return ax

    @plotting_context
    def monthly_return_dist(self, ax=None, display=True, **kwargs):
        if ax is None:
            ax = plt.gca()

        monthly_returns = self.returns.resample('1M').sum() * 100

        x_axis_formatter = FuncFormatter(utils.percentage)
        ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))

        ax.set_label('')
        ax.set_ylabel('Number of Months')
        ax.set_xlabel('Returns (%)')
        ax.set_title('Distribution of Monthly Returns', size = 14)

        monthly_returns.plot(ax=ax, kind='hist', bins=20,
                             alpha=0.8, color='orangered', **kwargs)
        ax.axvline(0.0, color='black', linestyle='-', lw=1.5, alpha=0.75)

        ax.axvline(
            monthly_returns.mean(),
            color='gold',
            linestyle='--',
            lw=4,
            alpha=1.0)

        ax.legend(['mean'],
                  loc='best')

        if display:
            plt.show()

        return ax

    def plot_annual_returns(self, ax=None, display=True, **kwargs):
        if ax is None:
            ax = plt.gca()

        annual_returns = pd.DataFrame(self.aggregate_returns('annual')) * 100

        x_axis_formatter = FuncFormatter(utils.percentage)
        ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))

        annual_returns.sort_index(ascending=False).plot(
            ax=ax, kind='barh', alpha=0.7, **kwargs)
        ax.axvline(
            annual_returns.values.mean(),
            color='steelblue',
            linestyle='--',
            lw=4,
            alpha=0.7)

        ax.legend(['mean'], loc='best')
        ax.set_ylabel('Year')
        ax.set_xlabel('Returns')
        ax.set_title('Annual Returns', size = 14)
        if display:
            plt.show()
        return ax

    def create_returns_tear_sheet(self, **kwargs):

        print("Entire data start date: %s" %
              self.returns.index[0].strftime('%Y-%m-%d'))
        print("Entire data end date: %s" %
              self.returns.index[-1].strftime('%Y-%m-%d'))
        print('\n')

        self.results_table()  # Print all performance stats

        vertical_sections = 8

        fig = plt.figure(figsize=(14, vertical_sections * 6))
        gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)

        ax_rolling_returns = plt.subplot(gs[:2, :])

        i = 2
        ax_rolling_returns_vol_match = plt.subplot(gs[i, :],
                                                   sharex=ax_rolling_returns)
        i += 1
        ax_rolling_returns_log = plt.subplot(gs[i, :],
                                             sharex=ax_rolling_returns)
        i += 1
        ax_returns = plt.subplot(gs[i, :],
                                 sharex=ax_rolling_returns)
        # i += 1
        # ax_rolling_beta = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
        i += 1
        ax_rolling_sharpe = plt.subplot(gs[i, :], sharex=ax_rolling_returns)

        i += 1
        ax_rolling_sortino = plt.subplot(gs[i, :],
                                 sharex=ax_rolling_returns)
        i += 1
        ax_monthly_heatmap = plt.subplot(gs[i, 0])
        ax_annual_returns = plt.subplot(gs[i, 1])
        ax_monthly_dist = plt.subplot(gs[i, 2])

        self.plot_cumulative_returns(ax=ax_rolling_returns,
                                     display=False)
        ax_annual_returns.set_title('Cumulative Returns')

        self.plot_cumulative_returns(ax=ax_rolling_returns_vol_match,
                                     volatility_match=True,
                                     display=False,
                                     )
        ax_rolling_returns_vol_match.set_title(
            'Cumulative returns volatility matched to benchmark')

        self.plot_cumulative_returns(ax=ax_rolling_returns_log,
                                     log_returns=True,
                                     display=False,
                                     )
        ax_rolling_returns_log.set_title(
            'Cumulative returns on logarithmic scale')

        self.plot_returns(ax=ax_returns,
                          display=False)

        self.plot_rolling_sharpe(ax=ax_rolling_sharpe,
                                 display=False)

        self.plot_rolling_sortino(ax=ax_rolling_sortino,
                                 display=False)

        self.plot_monthly_returns_heatmap(ax=ax_monthly_heatmap, display=False)

        self.plot_annual_returns(ax=ax_annual_returns, display=False)

        self.monthly_return_dist(ax=ax_monthly_dist, display=False)
        for ax in fig.axes:
            plt.setp(ax.get_xticklabels(), visible=True)

        plt.show()

        return fig

    def return_tears_data(self, display = False, to_excel = False, filename = None):
            
        # A list to store all results to export to excel
        results_list = ['basic_results_table', 'cumulative_returns', 'vol_matched_returns', 'rolling_sharpe', 'rolling_sortino',
                     'annual_returns', 'monthly_returns', 'returns_table']

        rolling_sharpe  = self.returns.rolling(
        126).apply(lambda x: self.sharpe(x))

        rolling_sortino = self.returns.rolling(
        126).apply(lambda x: self.sortino(x))

        # Total returns per year
        annual_returns = pd.DataFrame(self.aggregate_returns('annual')) * 100

        # Distribution of monthly returns
        monthly_returns = self.returns.resample('1M').sum() * 100

        cumulative_returns = self.cumulative_returns(starting_val=1)

        # Cumulative returns scaled to the benchmark volatility
        if self.benchmark is not None:
            bench_vol = self.benchmark.loc[self.returns.index].std()
            vol_matched_returns = (self.returns / self.returns.std()) * bench_vol
            vol_matched_returns = self.cumulative_returns(vol_matched_returns, starting_val = 1)
        
        # Returns table showing the return for each month and year.
        returns_table = self.aggregate_returns(MONTHLY)
        returns_table = returns_table.unstack().round(3)

        # Table of all basic statistics. Sharpe, cagr, drawdown etc.
        basic_results_table = self.results_table(display)
        
        results_dict = {}
        if to_excel:
            writer = ExcelWriter(filename)
            for x in results_list:
                stat = eval(x)
                if type(stat) == pd.Series:
                    stat.name = x
                    stat = stat.to_frame()
                stat.to_excel(writer, x)

            writer.close()
       
        for x in results_list:
            try:
                results_dict[x] = eval(x)        
            except:
                pass
        return results_dict
            
BASIC_STATS = ['cumulative_returns',
               'sharpe',
               'volatility',
               'cagr',
               'max_drawdown',
               'kurtosis',
               'skew',
               'sortino',
               'beta',
               'alpha',
               'tail_ratio',
               'stability_of_timeseries'
               ]
