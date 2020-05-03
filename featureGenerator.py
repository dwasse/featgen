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


def add_raw_features(df, period_dict, freq=None):
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
            col = get_col_name(feature, period, freq)
            roll_freq = str(period) + 's'
            if feature == 'volume':
                df[col] = df['volume'].rolling(roll_freq).sum()
            elif feature == 'vwap':
                df[col] = df['total'].rolling(roll_freq).sum() / df['volume'].rolling(roll_freq).sum()
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
            col = get_col_name(feature, period, freq)
            if feature == 'local_min_max':
                vwap_change_col = get_col_name('vwap_change', period=period, freq=freq)
                assert vwap_change_col in df
                df[col] = np.sign(np.sign(df[vwap_change_col]).diff())
            elif feature == 'max_change':
                max_col = get_col_name('max', period=period, freq=freq)
                assert get_col_name('max', period=period, freq=freq) in df
                df[col] = 100 * ((df[max_col] / df['price'].shift(period, freq='S')) - 1)
            elif feature == 'min_change':
                min_col = get_col_name('min', period=period, freq=freq)
                assert min_col in df
                df[col] = 100 * ((df[min_col] / df['price'].shift(period, freq='S')) - 1)
            elif feature == 'volatility':
                price_change_col = get_col_name('price_change', period=period, freq=freq)
                assert price_change_col in df
                df[col] = df[price_change_col].rolling(roll_freq).std()
    for feature in period_dict:
        for period in period_dict[feature]:
            col = get_col_name(feature, period, freq)
            if feature == 'volatility_change':
                vol_col = get_col_name('volatility', period, freq)
                assert vol_col in df
                df[col] = 100 * ((df[vol_col] / df[vol_col].shift(period, freq='S')) - 1)
    del df['total']
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)


def add_moving_avgs(df, period_dict, freq):
    """
    Add simple and/or exponentially weighted moving averages to dataframe.
    :param df: dataframe with datetime index
    :param period_dict: a dict with feature strings as keys and int periods as values (in seconds)
    :return:
    """
    for feature in period_dict:
        if feature == 'sma':
            add_sma(df, feature, period_dict[feature], freq)
        elif feature == 'ema':
            add_ema(df, feature, period_dict[feature], freq)


def add_sma(df, input_feature, periods, freq=None):
    """
    Add simple moving average of a given feature, over given periods.
    :param df: dataframe with datetime index
    :param input_feature: feature to compute sma over
    :param periods: list of integers
    :return:
    """
    for period in periods:
        col = get_col_name(input_feature + "_sma", period, freq)
        df[col] = df[input_feature].rolling(period).mean()


def add_ema(df, input_feature, periods, freq=None):
    """
    Add exponentially weighted moving average of a given feature, over given periods.
    :param df: dataframe with datetime index
    :param input_feature: feature to compute ema over
    :param periods: list of integers
    :return:
    """
    for period in periods:
        col = get_col_name(input_feature + "_ema", period, freq)
        df[col] = df[input_feature].ewm(span=period).mean()


def add_indicators(df, freq=None):
    """
    This function calculates technical analysis-based features on a dataframe.
    :param df: dataframe with datetime index
    """
    assert [c in ['open', 'high', 'low', 'close', 'volume'] for c in df]
    old_cols = list(df.columns)
    df = ta.add_all_ta_features(df, 'open', 'high', 'low', 'close', 'volume') \
        .fillna(method='ffill').fillna(0)
    new_cols = [c for c in list(df.columns) if c not in old_cols]
    col_dict = {c: get_col_name(feature=c,freq=freq) for c in new_cols}
    df.rename(columns=col_dict, inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df


# def add_s_r(df, periods, max_levels=10, freq='1h'):
#     from .supportResistance import RawPriceClusterLevels
#     """
#     Add support and resistance levels using Agglomerative Clustering.
#     :param df: dataframe with datetime index
#     :param periods: a list of lookback periods for support/resistance levels
#     :param max_levels: max number of support/resistance levels to add
#     :param freq: resampling frequency to speed up calculation (i.e. '1d')
#     :return: 
#     """
#     for period in periods:
#         resampled_df = df['close'].resample(freq).ohlc().fillna(method='ffill')
#         resampled_df.columns = [c.lower() for c in resampled_df]
#         for i in range(1, max_levels + 1):
#             resampled_df['s_r_level_' + str(i) + '_' + str(period)] = np.NaN
#             resampled_df['s_r_weight_' + str(i) + '_' + str(period)] = np.NaN
#         resampled_df['index'] = df.index
#         resampled_df.reset_index(inplace=True)
#         for index, row in resampled_df.iterrows():
#             cl = RawPriceClusterLevels(None, merge_percent=0.5, use_maximums=True, bars_for_peak=11)
#             start = index - period
#             if start < 0:
#                 start = 0
#             s_r_data = resampled_df.iloc[start:index]
#             cl.fit(s_r_data)
#             levels = cl.levels
#             if levels is not None:
#                 levels.sort(key=lambda x: x['peak_count'], reverse=True)
#                 levels_to_apply = levels[:max_levels]
#                 for i in range(1, len(levels_to_apply) + 1):
#                     resampled_df.loc[index, 's_r_level_' + str(i) + '_' + str(period)] = levels_to_apply[i - 1]['price']
#                     resampled_df.loc[index, 's_r_weight_' + str(i) + '_' + str(period)] = levels_to_apply[i - 1]['peak_count']
#         resampled_df.index = resampled_df['index']
#         del resampled_df['index']
#         for col in [c for c in df if 's_r' in c]:
#             df[col] = resampled_df[col]
#         df.fillna(method='ffill', inplace=True)
#         for i in range(1, max_levels + 1):
#             col = 's_r_level_' + str(i) + '_' + str(period)
#             if col in df:
#                 df['s_r_diff_' + str(i) + '_' + str(period)] = df['price'] - df[col]
#                 df['s_r_pct_' + str(i) + '_' + str(period)] = 100 * ((df['price'] / df[col]) - 1)


def get_col_name(feature, period=None, freq=None):
    if period is None:
        if freq is None:
            return feature
        return feature + "-" + freq
    if freq is None:
        return feature + "_" + str(period)
    return feature + "_" + str(period) + "-" + str(freq)


class FeatureGenerator:
    def __init__(self, config=config):
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
            print("Calculating features for frequency: %s" % freq)
            assert [c in df for c in ['open', 'high', 'low', 'close']]
            resampled_df = df.resample(freq).agg({'open': 'first',
                                                  'high': 'max',
                                                  'low': 'min',
                                                  'close': 'last'})
            resampled_df['volume'] = df['volume'].resample(freq).sum()
            # TODO: Should this be mean or most recent values?
            resampled_df['price'] = df['close'].resample(freq).mean()
            resampled_df.fillna(method='ffill', inplace=True)
            add_timestamp(resampled_df)

            # Calculate features
            if freq in self.config.raw_freqs:
                print("Calculating raw features...")
                add_raw_features(resampled_df, period_dict=self.config.raw_periods, freq=freq)
            if freq in self.config.moving_avg_freqs:
                print("Calculating moving avg features...")
                add_sma(resampled_df, 'price', self.config.moving_avg_periods['sma'], freq=freq)
                add_ema(resampled_df, 'price', self.config.moving_avg_periods['ema'], freq=freq)
            if freq in self.config.indicator_freqs:
                print("Calculating indicator features...")
                resampled_df = add_indicators(resampled_df, freq=freq)
            del resampled_df['timestamp']

            # Merge the resampled_df back into the original frequency
            for col in resampled_df:
                if col not in df:
                    df[col] = resampled_df[col]
        df.fillna(method='ffill', inplace=True)
