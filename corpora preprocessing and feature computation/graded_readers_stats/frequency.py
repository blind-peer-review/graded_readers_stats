##############################################################################
#                            FREQUENCY CALCULATIONS                          #
##############################################################################

from graded_readers_stats._typing import *
from graded_readers_stats.constants import *


def count_vocab_in_texts_grouped_by_level(
        phrases: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column: str,
        is_context: bool = False
) -> (str, Series):
    results = []
    for group_name in sentences_by_groups.groups:
        ctx_name = ' ' + CONTEXT + ' ' if is_context else ' '
        in_column_locations = column + ctx_name + LOCATIONS
        out_column_counts = column + ctx_name + COUNTS + ' ' + group_name

        counts = phrases.apply(
            lambda x: count_phrase_occurrences_v1(
                x[in_column_locations],
                sentences_by_groups.get_group(group_name).index
            ),
            axis=1
        )
        results.append((out_column_counts, counts))
    return results


def count_phrase_occurrences_v1(
        locations: [[(int, (int, int))]],  # list of docs of sentences
        group_indices: pd.Index
) -> int:
    # We count sentences since we detect the first occurrence of the vocab
    # meaning vocab appear at most once in a sentence.
    # NOTE: We are using '_' to indicate that we don't need the variable.
    return sum(1
               # iterate over pandas' series to recover index
               for doc_index in group_indices
               # we count sentences since we detect only
               # the first occurrence of the vocab in a sentence.
               # iterate over sentences in document
               for _ in locations[doc_index])


def frequency(
        locations: [[(int, (int, int))]],  # list of docs of sentences
) -> float:
    docs = locations
    return sum(1 for _ in docs)


def total_count_in_texts_grouped_by_level(
        vocabs: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column: str,
        is_context: bool = False
):
    results = []
    for group_name in sentences_by_groups.groups:
        ctx_name = ' ' + CONTEXT + ' ' if is_context else ' '
        out_column_counts = column + ctx_name + TOTAL_COUNTS + ' ' + group_name

        from pandas.core.common import flatten

        lemmas = sentences_by_groups.get_group(group_name)[COL_LEMMA]
        total_count = sum(1 for _ in flatten(lemmas))
        totals = vocabs.apply(lambda x: total_count, axis=1)
        results.append((out_column_counts, totals))
    return results


@DeprecationWarning
def total_counts_for_vocab(
        locations: [[(int, (int, int))]],
        texts: Series  # (2, [['Text', 'items']], ...)
) -> int:

    # Print the whole document (all sentences in doc)
    # print(
    #     [[w for sent in sents for w in sent]
    #      for doc_index, sents in texts.items()
    #      if len(locations[doc_index]) > 0]
    # )

    # Count number of words in a sentence.
    # Shape: [[1, 3, 1]], thus we need to unwrap outer list.
    # When there are no occurrences, we deal with an empty list [].
    total = [[len(sent) for sent in doc_sents]

             # iterate over pandas' series to recover index
             for doc_index, doc_sents in texts.items()

             # only docs with occurrences
             if len(locations[doc_index]) > 0]

    # sum word counts for all documents
    return sum(total[0]) if len(total) > 0 else 0


def frequency_in_texts_grouped_by_level(
        vocabs: DataFrame,
        sentences_by_groups: DataFrameGroupBy,
        column: str,
        is_context: bool = False
) -> (str, Series):
    results = []
    for group_name in sentences_by_groups.groups:
        ctx_name = ' ' + CONTEXT + ' ' if is_context else ' '
        column_count = column + ctx_name + COUNTS + ' ' + group_name
        column_total_count = column + ctx_name + TOTAL_COUNTS + ' ' + group_name
        column_frequency = column + ctx_name + FREQUENCY + ' ' + group_name

        freqs = vocabs.apply(
            lambda x: x[column_count] / x[column_total_count]
            if x[column_total_count] > 0 else 0,
            axis=1
        )
        results.append((column_frequency, freqs))
    return results


def count_doc_terms(docs_locs, term_indices):
    return [
        sum(len(doc_locs[index]) for index in term_indices)
        for doc_locs in docs_locs
    ]


def count_terms(terms_locs):
    return [sum(1 for sents in term_locs for _ in sents)
            for term_locs in terms_locs]


def context_counts_per_doc(locations_by_docs):
    return [  # list of docs
        [  # list of counts per context word
            sum(1 for _ in {loc for locs in context for loc in locs})
            for context in doc.values()
        ]
        for doc in locations_by_docs
    ]
    # Note: d = doc, c = context word, t = term


def freqs_by_term(terms_counts, words_total):
    return [count / words_total for count in terms_counts]
