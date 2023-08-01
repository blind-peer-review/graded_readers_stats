import time

from codetiming import Timer
from pandas.core.common import flatten

from graded_readers_stats import utils
from graded_readers_stats.constants import (
    COL_LEMMA,
    COL_LEVEL,
    COL_STANZA_DOC,
)
from graded_readers_stats.context import (
    collect_context_words_by_terms,
    freqs_pipeline,
    tfidfs_pipeline,
    trees_pipeline,
    locate_ctx_terms_in_docs,
    count_pipeline,
    avg,
)
from graded_readers_stats.data import read_pandas_csv
from graded_readers_stats.frequency import freqs_by_term, count_terms
from graded_readers_stats.preprocess import (
    run,
    vocabulary_pipeline,
    text_analysis_pipeline,
    locate_terms_in_docs,
)
from graded_readers_stats.tfidf import tfidfs
from graded_readers_stats.tree import terms_tree_props_pipeline


def analyze(args):
    vocabulary_path = args.vocabulary_path
    corpus_path = args.corpus_path
    level = args.level
    max_terms = args.max_terms
    max_docs = args.max_docs

    print()
    print('ANALYZE START')
    print('---')
    print('vocabulary_path = ', vocabulary_path)
    print('corpus_path = ', corpus_path)
    print('level = ', level)
    print('max_terms = ', max_terms)
    print('max_docs = ', max_docs)
    print('---')

    timer_text = '{name}: {:0.0f} seconds'
    start_main = time.time()

##############################################################################
#                                Preprocess                                  #
##############################################################################

    with Timer(name='Load data', text=timer_text):
        texts_df = read_pandas_csv(corpus_path)
        terms_df = read_pandas_csv(vocabulary_path)
        if max_terms:
            terms_df = terms_df[:max_terms]

    with Timer(name='Group', text=timer_text):
        texts_by_level = texts_df.groupby(COL_LEVEL)
        texts_df = texts_by_level.get_group(level).reset_index(drop=True)
        if max_docs:
            texts_df = texts_df[:max_docs]

    with Timer(name='Preprocess', text=timer_text):
        terms_df = run(terms_df, vocabulary_pipeline)
        texts_df = run(texts_df, text_analysis_pipeline)
        texts = texts_df[COL_LEMMA]
        storage = {
            'stanza': texts_df[COL_STANZA_DOC],
            'tree': {}
        }
        num_words = sum(1 for _ in flatten(texts))
        texts_df = texts_df.drop(columns=COL_STANZA_DOC)
        terms_df = terms_df.drop(columns=COL_STANZA_DOC)

##############################################################################
#                                 Terms                                      #
##############################################################################

    with Timer(name='Locate terms', text=timer_text):
        terms = [term for terms in terms_df[COL_LEMMA] for term in terms]
        terms_locs = locate_terms_in_docs(terms, texts)

    with Timer(name='Frequency', text=timer_text):
        terms_df['Count'] = terms_counts = count_terms(terms_locs)
        terms_df['Total'] = num_words
        terms_df['Frequency'] = freqs_by_term(terms_counts, num_words)

    with Timer(name='TFIDF', text=timer_text):
        terms_df['TFIDF'] = tfidfs(terms_locs, texts)

    with Timer(name='Tree', text=timer_text):
        terms_df['Tree'] = terms_tree_props_pipeline(storage, terms_locs)


##############################################################################
#                                Contexts                                    #
##############################################################################

    with Timer(name='Context collect', text=timer_text):
        ctx_words_by_term = collect_context_words_by_terms(terms_locs, texts, window=3)
        terms_df['Context words'] = ctx_words_by_term

    with Timer(name='Context locate terms', text=timer_text):
        ctxs_locs = locate_ctx_terms_in_docs(ctx_words_by_term, texts)

    with Timer(name='Context frequency', text=timer_text):
        terms_df['Context count per word'] = ctx_counts \
            = list(count_pipeline()(ctxs_locs))
        terms_df['Context count'] = list(map(avg, ctx_counts))
        terms_df['Context total'] = num_words
        terms_df['Context frequency'] \
            = list(freqs_pipeline(num_words)(ctx_counts))

    with Timer(name='Context TFIDF', text=timer_text):
        terms_df['Context TFIDF'] = list(tfidfs_pipeline(texts)(ctxs_locs))

    with Timer(name='Context Tree', text=timer_text):
        terms_df['Context tree'] = list(trees_pipeline(storage)(ctxs_locs))

##############################################################################
#                                  Others                                    #
##############################################################################

    with Timer(name='Export CSV', text=timer_text):
        terms_df = terms_df.drop(columns=[
            "Topic",
            "Subtopic",
            "Lemma",
            "Context words",
            "Context count per word"
        ])
        terms_df.to_csv(f'./output/terms_{level}.csv', index=False)

    print()
    utils.duration(start_main, 'Total time')
    print('')
    print('ANALYZE END')
