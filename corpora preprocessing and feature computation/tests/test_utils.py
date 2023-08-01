from graded_readers_stats.utils import first_occurrence_of_term_in_sent


def test_locate_simple_term():
    # Given
    hello = ['hello']
    world = ['world']
    universes = ['universes']
    #          0      1      2         3        4       5          6        7
    sent = ['hello', ',', 'world', 'merging', 'into', 'many', 'universes', '!']

    # When
    pos_hello = first_occurrence_of_term_in_sent(hello, sent)
    pos_world = first_occurrence_of_term_in_sent(world, sent)
    pos_universes = first_occurrence_of_term_in_sent(universes, sent)

    # Then
    assert (0, 1) == pos_hello
    assert (2, 3) == pos_world
    assert (6, 7) == pos_universes


def test_locate_multiword_term():
    # Given
    multiword1 = ['hello', ',', 'world']
    multiword2 = ['many', 'universes']
    #          0      1      2         3        4       5          6        7
    sent = ['hello', ',', 'world', 'merging', 'into', 'many', 'universes', '!']

    # When
    pos_mw1 = first_occurrence_of_term_in_sent(multiword1, sent)
    pos_mw2 = first_occurrence_of_term_in_sent(multiword2, sent)

    # Then
    assert (0, 3) == pos_mw1
    assert (5, 7) == pos_mw2


def test_locate_in_invalid_sent():
    # Given
    hello = ['hello']
    sent_none = [None]
    sent_num = [1]
    sent_tuple = [(1, 2)]

    # When
    pos_none = first_occurrence_of_term_in_sent(hello, sent_none)
    pos_num = first_occurrence_of_term_in_sent(hello, sent_num)
    pos_tuple = first_occurrence_of_term_in_sent(hello, sent_tuple)

    # Then
    assert pos_none is None
    assert pos_num is None
    assert pos_tuple is None
