import pytest

from graded_readers_stats.tree import *


# Non-test code is using dot syntax to access properties of classes.
# DotDict is for duck typing. Without DotDict, the tests wouldn't work.
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


@pytest.fixture
def storage():
    storage = {
        'tree': {},
        'stanza': [  # list of docs
            DotDict({  # a doc
                'sentences': [
                    # Head1
                    # |-- Word2
                    # |   +-- Word4
                    # +-- Word3
                    #     |-- Word5
                    #     +-- Word6
                    DotDict({'words': [  # sentence a.k.a. a sequence of words
                        DotDict({'id': 1, 'lemma': 'Head1', 'head': 0}),
                        DotDict({'id': 2, 'lemma': 'Word2', 'head': 1}),
                        DotDict({'id': 4, 'lemma': 'Word4', 'head': 2}),
                        DotDict({'id': 3, 'lemma': 'Word3', 'head': 1}),
                        DotDict({'id': 5, 'lemma': 'Word5', 'head': 3}),
                        DotDict({'id': 6, 'lemma': 'Word6', 'head': 3}),
                    ]}),

                    # Head1
                    # |-- Word2
                    # |   +-- Word4
                    # |   +-- Word7 NEW
                    # +-- Word3
                    #     |-- Word5
                    #     +-- Word6
                    #         +-- Word8 NEW
                    DotDict({'words': [
                        DotDict({'id': 1, 'lemma': 'Head1', 'head': 0}),
                        DotDict({'id': 2, 'lemma': 'Word2', 'head': 1}),
                        DotDict({'id': 4, 'lemma': 'Word4', 'head': 2}),
                        DotDict({'id': 7, 'lemma': 'Word7', 'head': 2}),
                        DotDict({'id': 3, 'lemma': 'Word3', 'head': 1}),
                        DotDict({'id': 5, 'lemma': 'Word5', 'head': 3}),
                        DotDict({'id': 6, 'lemma': 'Word6', 'head': 3}),
                        DotDict({'id': 8, 'lemma': 'Word8', 'head': 6}),
                    ]})
                ]
            })
        ]
    }
    return DotDict(storage)


def test_make_tree_for_loc(storage):
    # When
    tree = make_tree_for_loc(storage, 0, 0)

    # Then
    assert "Node('/Head1')" == repr(tree)
    assert "(Node('/Head1/Word2'), Node('/Head1/Word3'))" == repr(tree.children)
    assert "(Node('/Head1/Word2/Word4'), "\
           "Node('/Head1/Word3/Word5'), "\
           "Node('/Head1/Word3/Word6'))" == repr(tree.leaves)


def test_get_tree_props(storage):
    # Given
    docs_trees = [[
        make_tree_for_loc(storage, 0, 0),
        make_tree_for_loc(storage, 0, 1),
    ]]

    # When
    props = calc_mean_tree_props_for_nested_trees(docs_trees)

    # Then
    assert (3, 4, 3.5, 2, 3, 2.5) == props
