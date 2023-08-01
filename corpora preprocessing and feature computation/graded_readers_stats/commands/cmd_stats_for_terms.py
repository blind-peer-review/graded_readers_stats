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
from graded_readers_stats.data import read_pandas_csv
from graded_readers_stats.preprocess import run, vocabulary_pipeline
from graded_readers_stats.stats import calc_stats_for_group

timer_text = '{name}: {:0.0f} seconds'


def execute(args):
    vocabulary_path = args.vocabulary_path

    print()
    print('STATS START')
    print('---')
    print('vocabulary_path = ', vocabulary_path)
    print('---')

    start_main = time.time()
    terms_df = read_pandas_csv(vocabulary_path)
    terms_df = run(terms_df, vocabulary_pipeline)
    terms_df["multi-word"] = terms_df["Lemma"].apply(
        lambda x: len(list(flatten(x))) > 1
    )

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

    print()
    utils.duration(start_main, 'Total time')
    print('')
    print('STATS END')
