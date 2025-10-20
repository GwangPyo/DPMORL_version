import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, GroupByScaler, data_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


TRAIN_START_DATE = '2019-05-01'
TRAIN_END_DATE = '2024-12-31'
VALID_START_DATE = '2025-01-01'
VALID_END_DATE = '2025-04-31'
TEST_START_DATE = '2025-04-31'
TEST_END_DATE = '2025-09-30'

INDICATORS = [
    "macd",
    "rsi_7",
    "rsi_14",
    "cci_7",
    "cci_14",
    "close_10_ema",
    "close_5_ema",
    "close_20_ema",
    "close_50_ema",
    "close_100_ema",
]

asset_class = { 'Commodities': ['XLE', 'GLD', 'COMT'],
                'Bonds': ['TLT', 'TIP', 'JNK'],
                'Equities': ['SPY', 'QQQ', 'SOXX']
                }
tics = sum(list(asset_class.values()), [])


def download():
    df = YahooDownloader(start_date='2019-03-01',
                         end_date='2025-10-01',
                         ticker_list=tics).fetch_data()
    df_gold = YahooDownloader(start_date='2019-03-01',
                              end_date='2025-10-01',
                              ticker_list=['GC=F']).fetch_data()
    return df, df_gold


def load(path_df='./asset_df.csv', path_gold='./gold_df.csv'):
    """
    try:
        df = pd.read_csv(path_df)
        df_gold = pd.read_csv(path_gold)
    except FileNotFoundError:
    """
    df, df_gold = download()
    df.to_csv("asset_df.csv", index=False)
    df_gold.to_csv('gold_df.csv', index=False)
    return df, df_gold


if __name__ == '__main__':
    df, df_gold = load()
    print(df.tic.unique())

    gc_open = df_gold[["date", "open"]].rename(columns={ "open": "gc_open" })
    gc_open['gc_30_ema'] = gc_open['gc_open'].ewm(span=30, adjust=False).mean()
    gc_open['gc_30_ewmstd'] = gc_open['gc_open'].ewm(span=30, adjust=False).std(bias=False)

    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list=INDICATORS,
                         use_vix=True,
                         use_turbulence=False,
                         user_defined_feature=False)
    df = fe.preprocess_data(df)
    df = df.merge(gc_open, on="date", how="left")
    cols = ["open", "high", "low", "close"]
    for c in cols:
        # deflate and standardization to moving average and std
        df["gold_norm_" + c] = (df[c] - df['gc_30_ema']) / np.sqrt(df['gc_30_ewmstd'] * df["gc_open"])
    emas = ["close_10_ema", "close_5_ema", "close_20_ema", "close_50_ema", "close_100_ema"]

    for e in emas:
        df[e] = (df[e] - df['gc_30_ema']) / np.sqrt(df['gc_30_ewmstd'] * df["gc_open"])

    tech_indicators = ['macd', 'rsi_7', 'rsi_14',
                       'cci_7', "cci_14", 'volume',
                       'close_10_ema', 'close_5_ema', "close_50_ema",
                       'close_20_ema', 'close_100_ema', 'vix', 'gold_norm_open',
                       'gold_norm_high', 'gold_norm_low', 'gold_norm_close', 'gc_open']

    tech_indicator_val = df[tech_indicators].values
    # shift technical indicators so that a model cannot leverage future data
    df[tech_indicators] = np.roll(tech_indicator_val, axis=0, shift=1)
    df['volume'] = np.log1p(df['volume']) / 5
    df = df.rename(columns={ "volume": "log_volume",})

    train_df = data_split(df, TRAIN_START_DATE, TRAIN_END_DATE) # truncation happens. The leakage point (the first day) is truncated out.
    valid_df = data_split(df, VALID_START_DATE, VALID_END_DATE)
    test_df = data_split(df, TEST_START_DATE, TEST_END_DATE)

    train_df.to_csv("train_df.csv", index=True)
    print(train_df.isna().any())
    valid_df.to_csv("valid_df.csv", index=True)
    print(valid_df.isna().any())
    test_df.to_csv("test_df.csv", index=True)
    print(test_df.isna().any())