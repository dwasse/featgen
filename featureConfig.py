

day = 86400

# Periods in seconds
default_periods = [60, 300, 900, 1800, 3600, 3600*4, int(day/2), day]
raw_freqs = ['60s']
raw_periods = {
    'volume': default_periods,
    'vwap': default_periods,
    'volatility': default_periods,
    'volume_volatility': default_periods,
    'vwap_change': default_periods,
    'price_change': default_periods,
    'local_min_max': default_periods,
    'min': default_periods,
    'max': default_periods,
    'min_change': default_periods,
    'max_change': default_periods,
}

moving_avg_freqs = ['60s', '3600s', '21600s', '86400s']
moving_avg_periods = {
    'sma': [5, 10, 20, 50, 100],
    'ema': [5, 10, 20, 50, 100]
}

indicator_freqs = ['60s', '3600s', '21600s', '86400s']

s_r_freqs = ['6h']
s_r_periods = [120]

all_freqs = list(set(raw_freqs + moving_avg_freqs + indicator_freqs + s_r_freqs))

