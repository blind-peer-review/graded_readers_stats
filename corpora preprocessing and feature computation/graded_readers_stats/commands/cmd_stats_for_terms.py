from __future__ import annotations

import time

import matplotlib.pyplot as plt
import pandas as pd
from codetiming import Timer
from pandas.core.common import flatten

from graded_readers_stats import utils
from graded_readers_stats.constants import (
    COL_LEVEL,
)
from graded_readers_stats.data import read_pandas_csv, get_cache_path
from graded_readers_stats.preprocess import run, vocabulary_pipeline
from graded_readers_stats.stats import calc_stats_for_group
from graded_readers_stats.data import save_df_to_cache, load_df_from_pickle

timer_text = '{name}: {:0.0f} seconds'
cache_path = get_cache_path(__name__, 'vocabulary.pickle')


def execute(args):
    vocabulary_path = args.vocabulary_path

    print()
    print('STATS START')
    print('---')
    print('vocabulary_path = ', vocabulary_path)
    print('---')

    start_main = time.time()
    terms_df = load_df_from_pickle(cache_path)
    if terms_df.empty:
        terms_df = read_pandas_csv(vocabulary_path)
        terms_df = run(terms_df, vocabulary_pipeline)
        terms_df["multi-word"] = terms_df["Lemma"].apply(
            lambda x: len(list(flatten(x))) > 1
        )
        save_df_to_cache(terms_df, cache_path)

    groups = terms_df.groupby("Level")
    for name, df in groups:
        print(name)
        total = len(df)
        single = len(df.groupby("multi-word").get_group(False))
        multi = len(df.groupby("multi-word").get_group(True))
        print("Total:", len(df))
        print("Single:", single, f"({single/total*100:0.2f}%)")
        print("Multi:", multi, f"({multi/total*100:0.2f}%)")
        print()
    print('Total items:', len(terms_df))

    print()
    utils.duration(start_main, 'Total time')
    print('')
    print('STATS END')
