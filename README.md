This repository holds a feature generator for financial time series data. Given a dataframe with datetime index, data is resampled to given frequencies. Customize feature periods in `featureConfig.py`.

## Get started

Instantiate a FeatureGenerator object:
```
from .featureGenerator import FeatureGenerator
import featureConfig

fg = FeatureGenerator(featureConfig)
```

Pass in your dataframe:
```
df = pd.read_csv('example_data.csv')
fg.calculate_all_features(df)
```
Your dataframe now contains all desired features as new columns.

Additionally, you may access specific feature generation functions in a static context:
```
import .featureGenerator as featgen
df = pd.read_csv('example_data.csv')
periods = [10, 20, 30]
featgen.add_sma(df, 'close', periods)
```
`df` now contains 3 new columns, `close_sma_10`, `close_sma_20`, and `close_sma_30`.
