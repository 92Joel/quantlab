import pandas as pd
import numpy as np
import scipy.stats
import math
import pandas_datareader.data as web
import statsmodels.api as sm
from statsmodels import regression
import scipy.stats

from periods import ANNUALIZATION_FACTORS, APPROX_BDAYS_PER_YEAR
from periods import DAILY, WEEKLY, MONTHLY, ANNUAL
from . import utils

class Statistics(object):

    def __init__(self, returns, period = 'daily', benchmark=None, default_bench = True, risk_free=0):
        '''
        period:

        Optional string input. Accepts daily, weekly, monthly, annual.
        '''
        self.returns = returns
        self.benchmark = benchmark
        self.risk_free = risk_free
        # Periodicoty of returns for purposes of annualisation.
        self.period = period
        
        if self.benchmark is None and default_bench:
            start_date = self.returns.index[0]
            end_date = self.returns.index[-1]
            self.benchmark = utils.get_default_bench(start_date, end_date)

        if self.benchmark is not None:
            if returns.index[0] < self.benchmark.index[0]:
                self.returns = returns[returns.index >= self.benchmark.index[0]]

            if returns.index[0] > self.benchmark.index[0]:
                self.benchmark = self.benchmark[self.benchmark.index >= returns.index[0]]
        

    def sharpe(self, returns = None):
        ''' Period is the periodicity of the input returns.
            Defaults to DAILY
        '''
        ann_factor = ANNUALIZATION_FACTORS[self.period]

        # If returns are input to the method, then use those returns. Else use the returns initialised in the object constructor.
        if returns is None:
            returns = self.returns

        # if returns is not None:
        #     return np.sqrt(ann_factor) * (returns.mean() / returns.std()) 
        return np.sqrt(ann_factor) * (returns.mean() / returns.std()) 

    def cumulative_returns(self, returns = None, starting_val=0):
        if len(self.returns) < 1:
            return type(backtest_results)([])

        # If returns are input to the method, then use those returns. Else use the returns initialised in the object constructor.
        if returns is not None:
            rets = returns
        else:
            rets = self.returns.copy()

        if np.any(np.isnan(self.returns)):
            # print 'Filled', np.count_nonzero(np.isnan(self.returns)), 'nan
            # occurences in data with 0'
            rets.fillna(0, inplace=True)
        cum_df = (rets + 1).cumprod(axis=0)
        if starting_val == 0:
            return cum_df - 1
        else:
            return cum_df * starting_val

    def cagr(self):
        ''' Period is the periodicity of the input returns.
            Defaults to DAILY
        '''
        ann_factor = ANNUALIZATION_FACTORS[self.period]
        years = len(self.returns) / float(ann_factor)
        final = self.cumulative_returns(starting_val=1).iloc[-1]
        return final ** (1. / years) - 1

    def max_drawdown(self):
        values = self.cumulative_returns(starting_val=1).values
        i = np.argmax(np.maximum.accumulate(
            values) - values)  # end of the period
        j = np.argmax(values[:i])  # start of period
        date_start = self.returns.index[j]  # start of drawdown period
        date_end = self.returns.index[i]  # end of period
        # plt.plot(values)
        # plt.plot([date_end, date_start], [self.returns.iloc[i], self.returns.iloc[j]], 'o', color='Red', markersize=10)
        # plt.show()
        max_draw = ((values[i] - values[j]) / values[j]) 
        return max_draw

    def sortino(self, returns = None, target=0):
        ''' Period is the periodicity of the input returns.
            Defaults to DAILY
        '''

        # If returns are input to the method, then use those returns. Else use the returns initialised in the object constructor.
        if returns is None:
            returns = self.returns

        ann_factor = ANNUALIZATION_FACTORS[self.period]
        excess_daily_ret = returns - target
        target_downside_deviation = np.sqrt(
            np.mean(np.minimum(excess_daily_ret, 0.0)**2))
        sortino = excess_daily_ret.mean() / target_downside_deviation
        return sortino * np.sqrt(ann_factor)

    def alpha_beta(self, returns = None, factor_rets = None):
        if self.benchmark is None:
            "A benchmark input is required to calculate Alpha and Beta."
            return np.nan, np.nan
 
        # If returns are input to the method, then use those returns. Else use the returns initialised in the object constructor.
        if returns is None:
            returns = self.returns.copy()

        if factor_rets is None:
            factor_rets = self.benchmark.copy()

        if type(returns) == pd.Series or type(returns) == pd.DataFrame:
            returns = returns.values

        y = returns
        X = factor_rets
        if len(y) < 2 or len(X) < 2:
            print 'Not enough data. One of the input return series has length less than 2.'
            return np.nan, np.nan

        if len(y) != len(X):
            print 'Length mismatch of the two return series.'
            return np.nan, np.nan

        x = sm.add_constant(X)
        model = regression.linear_model.OLS(y, x).fit()
        alpha, beta = model.params[0], model.params[1]
        return alpha, beta

    def calmar_ratio(self):
        max_draw = self.max_drawdown()
        if max_draw >= 0:
            return np.nan
        else:
            return self.cagr() / np.abs(max_draw)

    def omega_ratio(self, return_required=0, period=APPROX_BDAYS_PER_YEAR):
        ''' Period is the period over which the expected return calculated.
        i.e. If a return of 100 is required over 1 year, this will equate to
        ~0.018 per day.
        '''
        return_threshold = (1 + return_required) ** \
            (1. / period) - 1
        returns_less_thresh = self.returns - self.risk_free - return_threshold
        returns_less_thresh = returns_less_thresh.values
        num = np.sum(returns_less_thresh[returns_less_thresh > 0])
        denom = -np.sum(returns_less_thresh[returns_less_thresh < 0])

        return num / denom

    def tail_ratio(self):
        '''
        Ratio of 95th percentile to the 5th.
        '''
        if type(self.returns) == pd.Series:
            returns = self.returns.values
        else:
            returns = np.asanyarray(self.returns)
        num = np.abs(np.nanpercentile(returns, 95))
        denom = np.abs(np.nanpercentile(returns, 5))
        return num / denom

    def volatility(self):
        '''
        Annualized volatility
        '''
        ann_factor = ANNUALIZATION_FACTORS[self.period]

        volatility = np.nanstd(self.returns, ddof=1) * np.sqrt((ann_factor))

        return volatility

    def aggregate_returns(self, convert_to):
        '''
        Aggregates returns by week, month, or year.
        Parameters
        ----------
        returns : pd.Series
        Daily returns of the strategy, noncumulative.
            - See full explanation in :func:`~empyrical.stats.cumulative_returns`.
        convert_to : str
            Can be 'weekly', 'monthly', or 'yearly'.
        Returns
        -------
        pd.Series
            Aggregated returns.
        '''

        def cumulate_returns(x):
            return self.cumulative_returns(x).iloc[-1]

        if convert_to == WEEKLY:
            grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
        elif convert_to == MONTHLY:
            grouping = [lambda x: x.year, lambda x: x.month]
        elif convert_to == ANNUAL:
            grouping = [lambda x: x.year]
        else:
            raise ValueError(
                'convert_to must be {}, {} or {}'.format(
                    WEEKLY, MONTHLY, YEARLY)
            )

        return self.returns.groupby(grouping).apply(cumulate_returns)

    def alpha(self, returns = None):
        alpha, beta = self.alpha_beta(returns)
        return alpha

    def beta(self, returns = None):
        alpha, beta = self.alpha_beta(returns)
        return beta
    
    def skew(self):
        return scipy.stats.skew(self.returns)

    def kurtosis(self):
        return scipy.stats.kurtosis(self.returns)

    def stability_of_timeseries(self):
        """
        Determines R-squared of a linear fit to the cumulative
        log returns. Computes an ordinary least squares linear fit,
        and returns R-squared.

        Parameters
        ----------
        returns : pd.Series or np.ndarray
            Daily returns of the strategy, noncumulative.
            - See full explanation in :func:`~empyrical.stats.cum_returns`.

        Returns
        -------
        float
            R-squared.

        """ 
        if len(self.returns) < 2:
            return np.nan

        returns = np.asanyarray(self.returns)
        returns = returns[~np.isnan(returns)]

        cum_log_returns = np.log1p(returns).cumsum()
        rhat = scipy.stats.linregress(np.arange(len(cum_log_returns)),
                            cum_log_returns)[2]

        return rhat ** 2

