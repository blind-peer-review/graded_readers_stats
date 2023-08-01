##############################################################################
#                            FREQUENCY CALCULATIONS                          #
##############################################################################
import math
from collections import Counter

from pandas.core.common import flatten

from graded_readers_stats._typing import *
from graded_readers_stats.constants import *


def tfidfs_for_groups(locs, doc_groups, column_id) -> pd.DataFrame:
    # Input: locations, documents
    # Output: a dataframe with TF-IDF columns ready to be merged

    # For each term in the vocabulary:

    # 0. Normalize
    # Convert join all texts and treat sentences as documents.
    # Requires averaging at the end to achieve a single value per term.

    # 1. Calculate the term frequency for each document
    # where a document is a sentence in the corpus.

    # 2. Calculate the inverse document frequency for each term
    # by dividing the number of documents (sents) in the group by level
    # by the number of documents where the term appears.

    # 3. Calculate the TF-IDF for each document.

    # 4. Average the TF-IDF for each document
    # and connect it with the term.

    result = {}
    # For each group:
    for group_name, group_df in doc_groups:

        # For each term in the vocabulary:
        term_result = []
        for term_locs in locs:

            # For each document in the group:
            tfidfs_group = []
            for doc_idx in group_df.index:
                doc_sents = group_df['Lemma'].loc[doc_idx]
                doc_matches = term_locs[doc_idx]

                sents_count = len(doc_sents)
                sents_matched = len(doc_matches)

                if sents_matched > 0:
                    idf_doc = math.log10(sents_count / sents_matched)

                    tfs_sents = []
                    for doc_match in doc_matches:
                        term_start = doc_match[1][0]
                        term_end = doc_match[1][1]
                        term_word_count = term_end - term_start
                        sent_idx = doc_match[0]
                        sent_words = doc_sents[sent_idx]
                        sent_word_count = len(sent_words)
                        # count multi-word terms as 1
                        # by removing its length and adding 1 instead
                        tf = 1 / (sent_word_count - term_word_count + 1)
                        tfs_sents.append(tf)

                    tfidfs_sents = map(lambda x: x * idf_doc, tfs_sents)
                    tfidf_doc_avg = sum(tfidfs_sents) / sents_count
                    tfidfs_group.append(tfidf_doc_avg)

            # TF-IDF for current term
            tfidf_group_avg = sum(tfidfs_group) / len(group_df)
            term_result.append(tfidf_group_avg)

        # save the averaged series of vocab terms tfidfs
        result[TFIDF + '_' + column_id + '_' + group_name] = term_result

    # a dataframe with columns by group ready to be merged
    # into the main dataframe
    return pd.DataFrame(result)


def tfidfs(terms_locs, docs):
    """
    term_locs
    ---
    A matrix with rows corresponding to terms and columns corresponding
    to docs. If there are 2 terms and 3 docs it would be a 2x3 matrix
    filled with indices of sentences where a term is found.
    Shape: [
        [[(sent_index, (term_start_index, term_end_index))]],
        [[...]],
        ...,
    ]

    docs
    ---
    docs is a list of doc where each
        doc is a list of sent where each
            sent is a list of word
    [
        [['This', 'is', 'a', 'sentence', '.'], ['And', 'this', 'as', 'well']],
        [['Another', 'sentence']]
    ]

    returns
    ---
    a list of TFIDF e.g. [0.0458, 0.0058, ...] where all the locations for a
    term are reduced to a single average TFIDF.

    """
    # For each term in the vocabulary:
    result_tfidfs_per_term = []
    for term_locs in terms_locs:

        tfidfs_docs = []
        for doc_idx, doc_sents in enumerate(docs):
            doc_matches = term_locs[doc_idx]

            sents_count = len(doc_sents)
            sents_matched = len(doc_matches)

            if sents_matched > 0:
                idf_doc = math.log10(sents_count / sents_matched)

                tfs_sents = []
                for doc_match in doc_matches:
                    term_start = doc_match[1][0]
                    term_end = doc_match[1][1]
                    term_word_count = term_end - term_start
                    sent_idx = doc_match[0]
                    sent_words = doc_sents[sent_idx]
                    # count multi-word terms as 1
                    # by removing its length and adding 1 instead
                    sent_word_count = (len(sent_words) - term_word_count + 1)
                    # use frequency 1 since we match 1 term per sentence max
                    tf = 1 / sent_word_count
                    tfs_sents.append(tf)

                tfidfs_sents = map(lambda tf: tf * idf_doc, tfs_sents)
                tfidfs_doc_sum = sum(tfidfs_sents)
                tfidf_doc_avg = tfidfs_doc_sum / sents_count
                tfidfs_docs.append(tfidf_doc_avg)

        # TF-IDF for current term
        tfidf_term_avg = sum(tfidfs_docs) / len(docs)
        result_tfidfs_per_term.append(tfidf_term_avg)

    return result_tfidfs_per_term


def calc_mean_doc_idfs(docs_terms_locs, term_indices):
    doc_matches_mask = [
        [1 if len(locs) else 0 for locs in [terms_locs[term_idx]
                                            for term_idx in term_indices]]
        for terms_locs in docs_terms_locs
    ]
    # [1, 1, 0, 1, 1]
    # [0, 0, 1, 1, 0]
    docs_match_counts = [
        sum(doc_matches)
        for doc_matches in doc_matches_mask
    ]
    # [4, 2]
    term_match_counts = [sum(column)
                         for column in zip(*doc_matches_mask)]
    # [1, 1, 1, 2, 1]
    doc_count = len(docs_terms_locs)
    # 2
    term_idfs = [
        math.log10(doc_count / matched) if matched > 0 else 0
        for matched in term_match_counts
    ]
    # [0.3010, 0.3010, 0.3010, 0.0000, 0.3010]
    docs_idfs = [
        [term_idfs[idx] if bit else 0
         for idx, bit in enumerate(doc_mask)]
        for doc_mask in doc_matches_mask
    ]
    # [
    #   [0.3010, 0.3010, 0, 0.0, 0.3010],
    #   [0, 0, 0.3010, 0.0, 0]
    # ]
    doc_avg_idfs = [
        sum(doc_idfs) / docs_match_counts[doc_idx]
        if docs_match_counts[doc_idx] > 0 else 0
        for doc_idx, doc_idfs in enumerate(docs_idfs)
    ]
    # [0.2257, 0.1505]
    return doc_avg_idfs

    # texts_df["IDF"] = calc_doc_avg_idfs(docs_locs)
    # texts_df['TFIDF'] = texts_df["Freq"] * texts_df["IDF"]
    # 0.03099
    # 0.01881


def calc_mean_doc_context_idfs(locations_by_docs):
    doc_contexts = [doc.keys() for doc in locations_by_docs]
    doc_count = len(locations_by_docs)
    context_counts = Counter(flatten(doc_contexts))
    context_idfs = {
        key: math.log10(doc_count / match_count) if match_count > 0 else 0
        for key, match_count in context_counts.items()
    }
    doc_avg_idfs = [
        sum(context_idfs[key] for key in context) / context_len
        if (context_len := len(context)) > 0 else 0
        for doc_idx, context in enumerate(doc_contexts)
    ]
    return doc_avg_idfs


