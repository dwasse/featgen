import numpy as np
import pandas as pd
import ta
import time
import logging
from . import featureConfig
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from mpl_finance import candlestick2_ohlc


def add_timestamp(df):
    """
    Add timestamp as a column for feature calculations.
    :param df: a dataframe with datetime index
    :return:
    """
    df['timestamp'] = df.index.astype(np.int64) // 10 ** 9


def add_raw_features(df, period_dict):
    """
    Add basic features to dataframe with given lookback periods.
    :param df: a dataframe with datetime index
    :param period_dict: a dict with feature strings as keys and int periods as values (in seconds)
    :return:
    """
    assert [c in df for c in ['volume', 'price']]
    df['total'] = df['volume'] * df['price']
    # Compute preliminary features
    for feature in period_dict:
        for period in period_dict[feature]:
            col = feature + '_' + str(period)
            roll_freq = str(period) + 's'
            if feature == 'volume':
                df[col] = df['volume'].rolling(roll_freq).sum()
            elif feature == 'vwap':
                df[col] = df['total'].rolling(roll_freq).sum() / df['volume'].rolling(roll_freq).sum()
            elif feature == 'volatility':
                df[col] = df['price'].rolling(roll_freq).std()
            elif feature == 'volume_volatility':
                df[col] = df['volume'].rolling(roll_freq).std()
            elif feature == 'vwap_change':
                df[col] = 100 * (((df['total'].rolling(roll_freq).sum() /
                                   df['volume'].rolling(roll_freq).sum()) /
                                   df['price'].shift(period, freq='S')) - 1)
            elif feature == 'price_change':
                df[col] = 100 * ((df['price'] / df['price'].shift(period, freq='S')) - 1)
            elif feature == 'max':
                df[col] = df['price'].rolling(roll_freq).max()
            elif feature == 'min':
                df[col] = df['price'].rolling(roll_freq).min()
    # Compute rest of features
    for feature in period_dict:
        for period in period_dict[feature]:
            col = feature + '_' + str(period)
            if feature == 'local_min_max':
                assert 'vwap_change' + '_' + str(period) in df
                df[col] = np.sign(np.sign(df['vwap_change_' + str(period)]).diff())
            elif feature == 'max_change':
                assert 'max' + '_' + str(period) in df
                df[col] = 100 * ((df['max_' + str(period)] / df['price'].shift(period, freq='S')) - 1)
            elif feature == 'min_change':
                assert 'min' + '_' + str(period) in df
                df[col] = 100 * ((df['min_' + str(period)] / df['price'].shift(period, freq='S')) - 1)
    del df['total']
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)


def add_moving_avgs(df, period_dict):
    """
    Add simple and/or exponentially weighted moving averages to dataframe.
    :param df: dataframe with datetime index
    :param period_dict: a dict with feature strings as keys and int periods as values (in seconds)
    :return:
    """
    for feature in period_dict:
        if feature == 'sma':
            add_sma(df, feature, period_dict[feature])
        elif feature == 'ema':
            add_ema(df, feature, period_dict[feature])


def add_sma(df, input_feature, periods):
    """
    Add simple moving average of a given feature, over given periods.
    :param df: dataframe with datetime index
    :param input_feature: feature to compute sma over
    :param periods: list of integers
    :return:
    """
    for period in periods:
        col = input_feature + '_sma_' + str(period)
        df[col] = df[input_feature].rolling(period).mean()


def add_ema(df, input_feature, periods):
    """
    Add exponentially weighted moving average of a given feature, over given periods.
    :param df: dataframe with datetime index
    :param input_feature: feature to compute ema over
    :param periods: list of integers
    :return:
    """
    for period in periods:
        col = input_feature + '_ema_' + str(period)
        df[col] = df[input_feature].ewm(span=period).mean()


def add_indicators(df):
    """
    This function calculates technical analysis-based features on a dataframe.
    :param df: dataframe with datetime index
    """
    assert [c in ['open', 'high', 'low', 'close', 'volume'] for c in df]
    df = ta.add_all_ta_features(df, 'open', 'high', 'low', 'close', 'volume') \
        .fillna(method='ffill').fillna(0)
    df.fillna(method='ffill', inplace=True)


def add_s_r(df, periods, max_levels=10, freq='1h'):
    from .supportResistance import RawPriceClusterLevels
    """
    Add support and resistance levels using Agglomerative Clustering.
    :param df: dataframe with datetime index
    :param periods: a list of lookback periods for support/resistance levels
    :param max_levels: max number of support/resistance levels to add
    :param freq: resampling frequency to speed up calculation (i.e. '1d')
    :return: 
    """
    for period in periods:
        resampled_df = df['close'].resample(freq).ohlc().fillna(method='ffill')
        resampled_df.columns = [c.lower() for c in resampled_df]
        for i in range(1, max_levels + 1):
            resampled_df['s_r_level_' + str(i) + '_' + str(period)] = np.NaN
            resampled_df['s_r_weight_' + str(i) + '_' + str(period)] = np.NaN
        resampled_df['index'] = df.index
        resampled_df.reset_index(inplace=True)
        for index, row in resampled_df.iterrows():
            cl = RawPriceClusterLevels(None, merge_percent=0.5, use_maximums=True, bars_for_peak=11)
            start = index - period
            if start < 0:
                start = 0
            s_r_data = resampled_df.iloc[start:index]
            cl.fit(s_r_data)
            levels = cl.levels
            if levels is not None:
                levels.sort(key=lambda x: x['peak_count'], reverse=True)
                levels_to_apply = levels[:max_levels]
                for i in range(1, len(levels_to_apply) + 1):
                    resampled_df.loc[index, 's_r_level_' + str(i) + '_' + str(period)] = levels_to_apply[i - 1]['price']
                    resampled_df.loc[index, 's_r_weight_' + str(i) + '_' + str(period)] = levels_to_apply[i - 1]['peak_count']
        resampled_df.index = resampled_df['index']
        del resampled_df['index']
        for col in [c for c in df if 's_r' in c]:
            df[col] = resampled_df[col]
        df.fillna(method='ffill', inplace=True)
        for i in range(1, max_levels + 1):
            col = 's_r_level_' + str(i) + '_' + str(period)
            if col in df:
                df['s_r_diff_' + str(i) + '_' + str(period)] = df['price'] - df[col]
                df['s_r_pct_' + str(i) + '_' + str(period)] = 100 * ((df['price'] / df[col]) - 1)


class FeatureGenerator:
    def __init__(self, config=featureConfig):
        self.config = config

    def calculate_all_features(self, df, freqs=None):
        """
        Calculate all features and insert them into a given dataframe.
        :param df: a dataframe with datetime index
        :param freqs: list of resampling frequencies for feature lookback periods
        :return:
        """
        if freqs is None:
            freqs = self.config.all_freqs
        for freq in freqs:
            assert [c in df for c in ['open', 'high', 'low', 'close']]
            resampled_df = df.resample(freq).agg({'open': 'first',
                                                  'high': 'max',
                                                  'low': 'min',
                                                  'close': 'last'})
            resampled_df['volume'] = df['volume'].resample(freq).sum()
            # TODO: Should this be mean or most recent values?
            resampled_df['price'] = df['close'].resample(freq).mean()
            resampled_df.fillna(method='ffill', inplace=True)

            # resampled_df = resampled_df.loc[~df.index.duplicated(keep='first')]

            add_timestamp(resampled_df)
            # Calculate features
            if freq in self.config.raw_freqs:
                add_raw_features(resampled_df, period_dict=self.config.raw_periods)
            if freq in self.config.moving_avg_freqs:
                # add_moving_avgs(resampled_df, period_dict=self.config.moving_avg_periods)
                add_sma(resampled_df, 'price', self.config.moving_avg_periods['sma'])
                add_ema(resampled_df, 'price', self.config.moving_avg_periods['ema'])
            if freq in self.config.indicator_freqs:
                add_indicators(resampled_df)
            if freq in self.config.s_r_freqs:
                add_s_r(resampled_df, periods=self.config.s_r_periods, freq=freq)
            del resampled_df['timestamp']
            # Merge the resampled_df back into the original frequency
            for col in resampled_df:
                if col not in df:
                    df[col] = resampled_df[col]
        df.fillna(method='ffill', inplace=True)










