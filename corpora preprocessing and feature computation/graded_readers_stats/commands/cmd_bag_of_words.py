import os.path
import time
from pathlib import Path

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
from graded_readers_stats.logit_tfidf import logit_tfidf
from graded_readers_stats.logit_word2vec import logit_word2vec
from graded_readers_stats.preprocess import (
    run,
    text_analysis_pipeline_ner,
    shrink_content_step,
)


def execute(args):
    corpus_path = args.corpus_path
    corpus_filename = Path(corpus_path).stem
    cache_path = f'./data/cache/cmd-bow-{corpus_filename}.pickle'
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
        texts_df_already_preprocessed = False
        if os.path.exists(cache_path):
            texts_df = pd.read_pickle(cache_path)
            texts_df_already_preprocessed = True
        else:
            texts_df = read_pandas_csv(corpus_path)
        if max_docs:
            texts_df = texts_df[:max_docs]

    with Timer(name='Preprocess', text=timer_text):
        if not texts_df_already_preprocessed:
            preprocessing_pipeline = text_analysis_pipeline_ner
            if shorten_content:
                preprocessing_pipeline.insert(1, shrink_content_step)
            texts_df = run(texts_df, preprocessing_pipeline)
            texts_df.to_pickle(cache_path)

        X_sents_raw = []
        y_sents = []
        for doc, level in zip(texts_df[COL_LEMMA], texts_df[COL_LEVEL]):
            X_sents_raw.extend(doc)
            y_sents.extend([level] * len(doc))

        if use_ner:
            # collect named entities from Stanza
            ner_items = texts_df[COL_STANZA_DOC]\
                .apply(lambda x: [e.text for e in x.ents])\
                .apply(lambda x: [w.lower() for w in x]) \
                .apply(lambda x: [w.split() for w in x]) \
                .apply(flatten) \
                .apply(list)
            ner_items_all = set(flatten(ner_items))

            # sentences
            X_sents_filtered = [[w for w in s if w not in ner_items_all] for s in X_sents_raw]
            X_sents = pd.Series([' '.join(s) for s in X_sents_filtered])

            # documents
            docs_flattened = texts_df[COL_LEMMA].apply(flatten).apply(list)
            docs_filtered = docs_flattened.apply(lambda doc: [w for w in doc if w not in ner_items_all])
            X = docs_filtered.apply(lambda x: ' '.join(x))
        else:
            # no processing, use "as-is"
            X_sents = pd.Series([' '.join(s) for s in X_sents_raw])
            X = texts_df[COL_LEMMA]\
                .apply(flatten)\
                .apply(list)\
                .apply(lambda x: ' '.join(x))

        y = texts_df[COL_LEVEL]
        y_sents = pd.Series(y_sents)

        # texts_df = texts_df.drop(columns=COL_STANZA_DOC, errors='ignore')

##############################################################################
#                             Logistic Regression
##############################################################################

    # with Timer(name='TF-IDF', text=timer_text):
    #     df = logit_tfidf(X, y)
    #     df.to_csv(f'./output/logit-bow-docs-{corpus_filename}.csv', index=False)
    #     df = logit_tfidf(X_sents, y_sents)
    #     df.to_csv(f'./output/logit-bow-sents-{corpus_filename}.csv', index=False)

    with Timer(name='Word2Vec', text=timer_text):
        df = logit_word2vec(X, y)
        df.to_csv(f'./output/logit-w2v-docs-{corpus_filename}.csv', index=False)
        df = logit_word2vec(X_sents, y_sents)
        df.to_csv(f'./output/logit-w2v-sents-{corpus_filename}.csv', index=False)

##############################################################################
#                                   Done
##############################################################################

    print()
    utils.duration(start_main, 'Total time')
    print('')
    print('BAG-OF-WORDS END')
