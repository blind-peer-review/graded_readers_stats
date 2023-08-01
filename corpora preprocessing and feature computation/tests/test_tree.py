import pandas as pd

from graded_readers_stats.constants import TREES_PROPS
from graded_readers_stats.deprecated.deprecated import \
    deprecated_calculate_tree_props
from graded_readers_stats.tree import *


def test_tree_props():
    # Given
    column = 'Reader_Avanzado_trees'
    data = {
        column: [[[
            make_tree(1, 3),
            make_tree(2, 4),
            make_tree(3, 5),
        ]]],
    }
    df = pd.DataFrame(data)

    # When
    new_column, tree_props = deprecated_calculate_tree_props(df, column)

    # Then
    assert column + TREES_PROPS == new_column
    assert (1, 3, 2, 3, 5, 4) == tree_props[0]


def test_calculate_tree_props_for_doc_contexts():
    # Given
    trees = [
        [  # doc
            make_tree(1, 3), make_tree(1, 3),
            make_tree(2, 4), make_tree(2, 4),
            make_tree(3, 5), make_tree(3, 5),
        ],
        [
            make_tree(4, 6), make_tree(8, 10),
        ],
        [],
    ]

    # When
    tree_props = calculate_tree_props_for_doc_contexts(trees)

    # Then
    # min_w, max_w, avg_w, min_h, max_h, avg_h
    expected = [
        (1, 3, 2.0, 3, 5, 4.0),               # doc 0
        (4, 8, 6.0, 6, 10, 8.0),              # doc 1
        (None, None, None, None, None, None)  # doc 2
    ]

    assert expected == tree_props


def test_calc_mean_doc_context_tree():
    # Given
    trees = [
        # doc 1             doc 2
        # sent 1            sent 1            sent 2
        [[make_tree(1, 3)], [make_tree(2, 4), make_tree(3, 5)]],  # term 1
        [[make_tree(2, 2)], [make_tree(4, 4), make_tree(6, 6)]],  # term 2
        [[], [], []],
        [],
        #          ^ low            ^ avg            ^ high
    ]

    # When
    tree_props = calculate_tree_props_v2(trees)

    # Then
    # min_w, max_w, avg_w, min_h, max_h, avg_h
    expected = [
        (1, 3, 2, 3, 5, 4),
        (2, 6, 4, 2, 6, 4),
        (None, None, None, None, None, None),
        (None, None, None, None, None, None),
    ]
    assert expected == tree_props


def make_tree(width, height):
    assert width > 0
    assert height > 0

    height_origin = height + 1
    root = Node("root")
    head = root

    while height > 0:
        node = Node("level_" + str(height_origin - height))
        node.parent = head
        head = node
        height -= 1

    width -= 1  # take into account height node
    while width > 0:
        node = Node("level_1_" + str(width))
        node.parent = root
        width -= 1

    return root
