import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from mpl_finance import candlestick2_ohlc


class BaseLevelFinder:
    def __init__(self, merge_distance, merge_percent=None, level_selector='median'):

        self._merge_distance = merge_distance
        self._merge_percent = merge_percent

        self._level_selector = level_selector

        self._levels = None
        self._validate_init_args()

    @property
    def levels(self):
        return self._levels

    def _validate_init_args(self):
        pass

    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            X = data['close'].values
        elif isinstance(data, np.array):
            X = data
        else:
            raise Exception(
                'Only np.array and pd.DataFrame are supported in `fit` method'
            )

        prices = self._find_potential_level_prices(X)
        levels = self._aggregate_prices_to_levels(prices, self._get_distance(X))

        self._levels = levels

    def _find_potential_level_prices(self, X):
        raise NotImplementedError()

    def _get_distance(self, X):
        if self._merge_distance:
            return self._merge_distance

        mean_price = np.mean(X)
        return self._merge_percent * mean_price / 100

    def _aggregate_prices_to_levels(self, pivot_prices, distance):
        raise NotImplementedError()


def _cluster_prices_to_levels(prices, distance, level_selector='mean'):
    clustering = AgglomerativeClustering(distance_threshold=distance, n_clusters=None)
    try:
        clustering.fit(prices.reshape(-1, 1))
    except ValueError:
        return None

    df = pd.DataFrame(data=prices, columns=('price',))
    df['cluster'] = clustering.labels_
    df['peak_count'] = 1

    grouped = df.groupby('cluster').agg(
        {
            'price': level_selector,
            'peak_count': 'sum'
        }
    ).reset_index()

    return grouped.to_dict('records')


class RawPriceClusterLevels(BaseLevelFinder):
    def __init__(self, merge_distance, merge_percent=None, level_selector='median', use_maximums=True,
                 bars_for_peak=21):

        self._use_max = use_maximums
        self._bars_for_peak = bars_for_peak
        super().__init__(merge_distance, merge_percent, level_selector)

    def _validate_init_args(self):
        super()._validate_init_args()
        if self._bars_for_peak % 2 == 0:
            raise Exception('N bars to define peak should be odd number')

    def _find_potential_level_prices(self, X):
        d = pd.DataFrame(data=X, columns=('price',))
        bars_to_shift = int((self._bars_for_peak - 1) / 2)

        if self._use_max:
            d['F'] = d['price'].rolling(window=self._bars_for_peak).max().shift(-bars_to_shift)
        else:
            d['F'] = d['price'].rolling(window=self._bars_for_peak).min().shift(-bars_to_shift)

        prices = pd.unique(d[d['F'] == d['price']]['price'])

        return prices

    def _aggregate_prices_to_levels(self, prices, distance):
        return _cluster_prices_to_levels(prices, distance, self._level_selector)


def _plot_levels(where, levels, only_good=False):
    for l in levels:

        if isinstance(l, float):
            where.axhline(y=l, color='black', linestyle='-')
        elif isinstance(l, dict):
            if 'score' in l.keys():
                if only_good and l['score'] < 0:
                    continue
                color = 'red' if l['score'] < 0 else 'blue'
                where.axhline(y=l['price'], color=color, linestyle='-', linewidth=0.2 * abs(l['score']))
            else:
                where.axhline(y=l['price'], color='black', linestyle='-')


def plot_levels_on_candlestick(df, levels, only_good=False, path=None):
    df['time'] = df.index
    ohlc = df[['time', 'open', 'high', 'low', 'close']].copy()
    ohlc["time"] = pd.to_datetime(ohlc['time'])
    ohlc["time"] = ohlc["time"].apply(lambda x: mdates.date2num(x))
    del df['time']
    f1, ax = plt.subplots(figsize=(10, 5))
    candlestick2_ohlc(ax,
                      closes=ohlc.close.values,
                      opens=ohlc.open.values,
                      highs=ohlc.high.values,
                      lows=ohlc.low.values,
                      colordown='red',
                      colorup='green'
                      )

    _plot_levels(ax, levels, only_good)

    if path:
        plt.savefig(path)
    else:
        plt.show()