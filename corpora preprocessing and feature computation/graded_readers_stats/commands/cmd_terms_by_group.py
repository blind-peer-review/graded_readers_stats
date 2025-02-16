import time

import pandas as pd
from codetiming import Timer

from graded_readers_stats import utils
from graded_readers_stats.constants import COL_LEVEL, COL_LEMMA, COL_STANZA_DOC
from graded_readers_stats.data import read_pandas_csv
from graded_readers_stats.preprocess import vocabulary_pipeline, run, \
    text_analysis_pipeline_simple, locate_terms_in_docs


def analyze(args):
    vocabulary_path = args.vocabulary_path
    corpus_path = args.corpus_path
    max_terms = args.max_terms
    max_docs = args.max_docs

    print()
    print('ANALYZE START')
    print('---')
    print('vocabulary_path = ', vocabulary_path)
    print('corpus_path = ', corpus_path)
    print('max_terms = ', max_terms)
    print('max_docs = ', max_docs)
    print('---')

    timer_text = '{name}: {:0.0f} seconds'
    start_main = time.time()

##############################################################################
#                                Preprocess                                  #
##############################################################################

    result = {}
    terms_df = read_pandas_csv(vocabulary_path)
    if max_terms:
        terms_df = terms_df[:max_terms]
    terms_df = run(terms_df, vocabulary_pipeline)
    terms_df = terms_df.drop(columns=COL_STANZA_DOC, errors='ignore')
    terms = [term for terms in terms_df[COL_LEMMA] for term in terms]

    for cur_path in corpus_path:
        with Timer(name='Load data', text=timer_text):
            texts_df = read_pandas_csv(cur_path)
            if max_docs:
                texts_df = texts_df[:max_docs]

        with Timer(name='Preprocess', text=timer_text):
            texts_df = run(texts_df, text_analysis_pipeline_simple)
            texts_df = texts_df.drop(columns=COL_STANZA_DOC, errors='ignore')

        with Timer(name='Locate terms', text=timer_text):
            texts = texts_df[COL_LEMMA]
            terms_docs_locs = locate_terms_in_docs(terms, texts)

        with Timer(name='Term group membership', text=timer_text):
            groups = texts_df.groupby(COL_LEVEL)
            for group_name, group_df in groups:
                result[group_name] = [0] * len(terms_df)
                result[group_name] = [sum(1 for doc_idx, _ in group_df.iterrows()
                                          for _ in docs_locs[doc_idx])
                                      for docs_locs in terms_docs_locs]

    with Timer(name='Export CSV', text=timer_text):
        terms_df = terms_df.join(pd.DataFrame(result))
        terms_df = terms_df.drop(columns=["Topic", "Subtopic", "Lemma"], errors='ignore')
        terms_df.to_csv(f'./output/term_counts_by_text_group.csv', index=False)

    print()
    utils.duration(start_main, 'Total time')
    print('')
    print('ANALYZE END')
