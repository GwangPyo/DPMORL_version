from finrl_env.mo_envs import StockTradingMOEnv
from typing import Literal
import pandas as pd
from gymnasium.wrappers import RescaleAction


# Seed shuffling prime numbers
PRIME_1 = 8610829
PRIME_2 = 8572451
PRIME_3 = 1497233


def load_env(mode: Literal['train', 'valid', 'test'] = 'train', seed: int = 42):
    path = { "train": 'train_df.csv', 'test': 'test_df.csv' }
    options = { "train": { "randomize_day": True,
                           "bidding": "adv_uniform",
                           "stop_loss_calculation": 'low',
                           "seed": seed,
                           },

                "test": { "randomize_day": False,
                          "bidding": "uniform",
                          "stop_loss_calculation": 'close',
                          "seed": (seed * PRIME_2) % int(2 ** 31)
                          }
                }

    df = pd.read_csv(path[mode])
    df = df.set_index(df.columns[0])
    return RescaleAction(StockTradingMOEnv(df, **(options[mode])), min_action=-1, max_action=1)
