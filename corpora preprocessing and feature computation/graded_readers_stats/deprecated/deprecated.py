##############################################################################
#                                Deprecated                                  #
##############################################################################

from anytree import Node
from stanza.models.common.doc import Sentence

from graded_readers_stats._typing import *
from graded_readers_stats.constants import *
from graded_readers_stats.tree import calc_mean_tree_props_for_nested_trees


def deprecated_make_trees_for_occurrences_v2(
        terms_locs: [],
        stanza_docs: Series
):
    return [deprecated_make_tree(term_locs, stanza_docs)
            for term_locs in terms_locs]


def deprecated_make_tree(
        docs_locs: [[(int, (int, int))]],  # list of docs of sents
        stanza_docs: Series
) -> [(int, Node)]:
    docs_trees = []
    for doc_index, doc_locs in enumerate(docs_locs):
        doc_trees = []
        for sent_loc in doc_locs:
            sent_index = sent_loc[0]
            sent = stanza_docs[doc_index].sentences[sent_index]
            root_node = deprecated_make_tree_for_sent(sent)
            doc_trees.append(root_node)
        docs_trees.append(doc_trees)
    return docs_trees


def deprecated_make_tree_for_sent(sent: Sentence) -> Node:
    nodes = {word.id: Node(word.lemma) for word in sent.words}
    for word in sent.words:
        nodes[word.id].parent = nodes[word.head] if word.head > 0 else None
        if word.head == 0:
            root = nodes[word.id]
    return root


def deprecated_make_trees_for_occurrences(
        vocab: DataFrame,
        texts: DataFrame,
        column: str
) -> (str, [[(int, Node)]]):
    in_column_locations = column + ' ' + LOCATIONS
    out_column_trees = column + ' ' + 'trees'

    rows = vocab[in_column_locations]
    trees = [deprecated_make_tree(row, texts[COL_STANZA_DOC]) for row in rows]
    return out_column_trees, trees


def deprecated_calculate_tree_props(
        vocabs: DataFrame,
        column: str
) -> (str, Series):
    in_column_trees = column
    out_column_trees_props = column + TREES_PROPS
    values = vocabs.apply(
        lambda x: calc_mean_tree_props_for_nested_trees(x[in_column_trees]),
        axis=1
    )
    return out_column_trees_props, values


