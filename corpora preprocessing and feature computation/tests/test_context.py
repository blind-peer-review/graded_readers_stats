import pytest

from graded_readers_stats.context import (
    collect_context_words_by_terms,
    collect_context_words_for_term
)


@pytest.fixture
def terms_locs():
    # term_locs are list of doc locs. A doc term can be empty if nothing is
    # found for a doc. A doc is a list of sentence locs. A sentence loc is a
    # tuple of sentence index and position of the terms inside the sentence.
    return [
        [  # term 1 = ['another']
            [(1, (2, 3))],  # doc 1
            [],             # doc 2
        ],
        [  # term 2 = ['beautiful']
            [],             # doc 1
            [(0, (1, 2))],  # doc 2
        ],
        [  # term 3 = ['is', 'it'] (multi-word term)
            [],             # doc 1
            [(1, (1, 3))],  # doc 2
        ],
    ]


@pytest.fixture
def docs():
    # docs are list of sentences which are list of words
    return [
        [  # doc 1
            ['this', 'is', 'a', 'sentence', '.'],  # sent 1
            ['this', 'is', 'another', 'sentence', '.'],  # sent 2
            ['all', 'things', 'are', 'happy', 'when', 'I', 'am', 'happy', '.'],
        ],
        [  # doc 2
            ['hello', 'beautiful', 'world', '!'],  # sent 1
            ['how', 'is', 'it', 'going', 'today', '?'],  # sent 2
        ],
    ]


def test_collect_context_single_term(terms_locs, docs):
    # Given
    another_locs = terms_locs[0]  # list of locations per doc

    # When
    words = collect_context_words_for_term(another_locs, docs, 1)

    # Then
    assert ['is', 'sentence'] == words


def test_collect_context_multiword_term(terms_locs, docs):
    # Given
    is_it_locs = terms_locs[2]  # list of locations per doc

    # When
    words = collect_context_words_for_term(is_it_locs, docs, 1)

    # Then
    assert ['how', 'going'] == words


def test_collect_context_window_param(terms_locs, docs):
    # Given
    is_it_locs = terms_locs[2]  # list of locations per doc

    # When
    words = collect_context_words_for_term(is_it_locs, docs, 2)

    # Then
    assert ['how', 'going', 'today'] == words


def test_collect_context_list_of_terms(terms_locs, docs):
    # When
    words = collect_context_words_by_terms(terms_locs, docs, 1)

    # Then
    expected = [
        ['is', 'sentence'],  # term1 context
        ['hello', 'world'],  # term2 context
        ['how', 'going']     # term3 context
]
    assert expected == words


