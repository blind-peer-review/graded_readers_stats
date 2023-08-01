import time

import pandas as pd
from codetiming import Timer
from pandas.core.common import flatten

from graded_readers_stats import utils
from graded_readers_stats.constants import (
    COL_LEMMA,
    COL_STANZA_DOC,
    COL_LEVEL,
)
from graded_readers_stats.data import read_pandas_csv
from graded_readers_stats.logit import logit
from graded_readers_stats.preprocess import (
    run,
    text_analysis_pipeline_ner,
    shrink_content_step,
)


def execute(args):
    corpus_path = args.corpus_path
    max_docs = args.max_docs
    shorten_content = args.shorten_content
    use_ner = args.strip_named_entities

    print()
    print('BAG-OF-WORDS START')
    print('---')
    print('corpus_path = ', corpus_path)
    print('max_docs = ', max_docs)
    print('---')

    timer_text = '{name}: {:0.0f} seconds'
    start_main = time.time()

##############################################################################
#                                Preprocess                                  #
##############################################################################

    with Timer(name='Load data', text=timer_text):
        texts_df = read_pandas_csv(corpus_path)
        if max_docs:
            texts_df = texts_df[:max_docs]

    with Timer(name='Preprocess', text=timer_text):
        text_analysis_pipeline = text_analysis_pipeline_ner
        if shorten_content:
            text_analysis_pipeline.insert(1, shrink_content_step)

        texts_df = run(texts_df, text_analysis_pipeline)
        if use_ner:
            ner_items = texts_df[COL_STANZA_DOC]\
                .apply(lambda x: [e.text for e in x.ents])\
                .apply(lambda x: [w.lower() for w in x])
            lemmas_dict = texts_df[COL_LEMMA]\
                .apply(flatten)\
                .apply(list)
            lemmas_series = pd.Series([
                list(filter(lambda x: x not in ner_items[index], lemma))
                for index, lemma in lemmas_dict.items()
            ])
            X = lemmas_series.apply(lambda x: ' '.join(x))
        else:
            X = texts_df[COL_LEMMA]\
                .apply(flatten)\
                .apply(list)\
                .apply(lambda x: ' '.join(x))
        y = texts_df[COL_LEVEL]
        # texts_df = texts_df.drop(columns=COL_STANZA_DOC)

##############################################################################
#                             Logistic Regression
##############################################################################

    with Timer(name='Logistic Regression', text=timer_text):
        df = logit(X, y)

    with Timer(name='Export CSV', text=timer_text):
        file_name = corpus_path.split("/")[-1]
        df.to_csv(f'./output/logit-bow-{file_name}', index=False)
##############################################################################
#                                   Done
##############################################################################

    print()
    utils.duration(start_main, 'Total time')
    print('')
    print('BAG-OF-WORDS END')
