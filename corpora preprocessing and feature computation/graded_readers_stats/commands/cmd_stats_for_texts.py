from __future__ import annotations

import time

import matplotlib.pyplot as plt
import pandas as pd
from codetiming import Timer

from graded_readers_stats import utils
from graded_readers_stats.constants import (
    COL_LEVEL,
)
from graded_readers_stats.data import read_pandas_csv
from graded_readers_stats.stats import calc_stats_for_group

timer_text = '{name}: {:0.0f} seconds'


def execute(args):
    corpus_paths = args.corpus_paths
    max_docs = args.max_docs

    print()
    print('STATS START')
    print('---')
    print('corpus_paths = ', corpus_paths)
    print('---')

    start_main = time.time()

    groups = [(group_name, group_df)
              for corpus_path in corpus_paths
              for group_name, group_df in load_groups(corpus_path, max_docs)]
    stats = {name: calc_stats_for_group(name, df, max_docs)
             for name, df in groups}

    sort_order = {"Inicial": 0, "Intermedio": 1, "Avanzado": 2}
    group_names = [name for name, _ in groups]
    group_names = sorted(group_names, key=lambda x: sort_order[x])

    # df = pd.DataFrame(
    #     {k: [stats[nn]["deprel"].setdefault(k, 0) for nn in group_names]
    #      for name in group_names
    #      for k in stats[name]["deprel"].keys()
    #      },
    #     index=group_names
    # )

    # ll = sorted(list(stats["Inicial"]["deprel"].keys()))
    # for i in range(0, len(ll), 3):
    #     df.plot(kind="bar", y=ll[i:3+i], subplots=True, figsize=(5, 15))
    #     plt.xticks(rotation=0, ha='right')
    #     # plt.savefig(f"output/stats/deprel-image-{i}.png")
    #     plt.show()

    print()
    utils.duration(start_main, 'Total time')
    print('')
    print('STATS END')


def load_groups(corpus_path: str, max_docs: int) -> list[pd.DataFrame]:

    with Timer(name=f'Load data', text=timer_text):
        texts = read_pandas_csv(corpus_path)

    with Timer(name=f'Group', text=timer_text):
        texts_groupedby = texts.groupby(COL_LEVEL)
        for group_name in texts_groupedby.groups:
            print("\n---")
            print("Processing group:", group_name)
            group_df = texts_groupedby\
                .get_group(group_name)\
                .reset_index(drop=True)\
                .copy()
            yield group_name, group_df


