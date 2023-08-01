import pandas as pd

from graded_readers_stats.context import tfidfs_pipeline
from graded_readers_stats.tfidf import tfidfs, tfidfs_for_groups, \
    calc_mean_doc_context_idfs


def test_tfidfs_for_groups():
    # Given
    vocab_d = {
               'Lexical item': [
                   'multi word 1',
                   'word1'
               ],
               'Locations': [
                   [[], [(0, (0, 3))], [], []],
                   [[(0, (0, 1))], [], [], [(0, (0, 1))]],
               ]}
    vocab_df = pd.DataFrame(vocab_d)

    docs_d = {
              'Level': [
                  'Inicial',
                  'Avanzado',
                  'Inicial',
                  'Avanzado'
              ],
              'Lemma': [
                  [['word1', 'word2', '.'],
                   ['word3', 'word4', '.'],
                   ['word5', 'word6', '.'],
                   ['word7', 'word8', '.']],
                  [['multi', 'word', '1', 'word9', 'word10', '.'],
                   ['word11', 'word12', 'word15', 'word16', '.']],
                  [['word3', 'word4', '.']],
                  [['word13', 'word14', '.']],
              ]}
    docs_df = pd.DataFrame(docs_d)

    # When
    result = tfidfs_for_groups(
        locs=vocab_df['Locations'],
        doc_groups=docs_df.groupby('Level'),
        column_id='Reader'
    )

    # Then
    prefix = 'TFIDF_Reader_'
    assert 0.0 == result[prefix + 'Inicial'][0]
    assert 0.025085832971998432 == result[prefix + 'Inicial'][1]
    assert 0.018814374728998825 == result[prefix + 'Avanzado'][0]
    assert 0.0 == result[prefix + 'Avanzado'][1]


def test_tfidf2():
    # Given
    vocab_d = {
        'Lexical item': [
            'multi word 1',
            'word1'
        ],
        'Locations': [
            [[], []],
            [[(0, (0, 1))], [(0, (0, 1))]],
        ]}
    vocab_df = pd.DataFrame(vocab_d)

    docs_d = {
        'Topic': [
            'Biology',
            'Medicine',
        ],
        'Lemma': [
            [['word1', 'word2', '.'],
             ['word3', 'word4', '.'],
             ['word5', 'word6', '.'],
             ['word7', 'word8', '.']],
            [['word1', 'word4', '.']],
        ]
    }
    docs_df = pd.DataFrame(docs_d)

    # When
    vocab_tfidfs = tfidfs(vocab_df['Locations'], docs_df['Lemma'])

    # Then
    assert 0.0 == vocab_tfidfs[0]
    assert 0.025085832971998432 == vocab_tfidfs[1]


def test_tfidf3():
    # Given
    vocab_d = {
        'Lexical item': [
            'multi word 1',
            'word1'
        ],
        'Locations': [
            [[(0, (0, 3))], []],
            [[], []],
        ]}
    vocab_df = pd.DataFrame(vocab_d)

    docs_d = {
        'Topic': [
            'Biology',
            'Medicine',
        ],
        'Lemma': [
            [['multi', 'word', '1', 'word9', 'word10', '.'],
             ['word11', 'word12', 'word15', 'word16', '.']],
            [['word13', 'word14', '.']],
        ]
    }
    docs_df = pd.DataFrame(docs_d)

    # When
    vocab_tfidfs = tfidfs(vocab_df['Locations'], docs_df['Lemma'])

    # Then
    assert 0.018814374728998825 == vocab_tfidfs[0]
    assert 0.0 == vocab_tfidfs[1]


# TODO: Doesn't work since sentences match only first occurrence
#  i.e. example would be matched only once even though appears 3 times.
# def test_tfidf4():
#     # Given
#     s = (0, (0, 1))
#     locations = [
#         [[s], [s]],  # this
#         [[], [s, s, s]],  # example
#     ]
#
#     docs = [  # 2 docs with 1 sentence each
#         [['this', 'is', 'a', 'a', 'sample']],
#         [['this', 'is', 'an', 'an', 'example', 'example', 'example']],
#     ]
#
#     # When
#     result = tfidfs(term_locs=locations, docs=docs)
#
#     # Then
#     expected = [
#         0,      # this
#         0.0645  # example = (0 + 0.129) / 2
#     ]
#     assert expected == result


def test_tfidf5():
    # Given
    s1 = (0, (0, 1))
    s2 = (1, (0, 1))
    locations = [
        [[s1, s2]],  # this
        [[s2]],  # example
    ]

    docs = [
        [['this', 'is', 'a', 'a', 'sample'],
         ['this', 'is', 'an', 'an', 'example', 'example', 'example']],
    ]

    # When
    result = tfidfs(terms_locs=locations, docs=docs)

    # Then
    expected = [
        0,                      # this
        0.021502142547427227    # example
    ]
    assert expected == result


def test_tfidfs_pipeline():
    # Given
    s = (0, (0, 1))  # some sentence occurrence
    terms_locs = [  # terms
        [  # term 1: 0.2
            [  # ctx word 1: (0.0301 + 0.0301 + 0) / 3 ~= 0.2
                [s],  # doc 1: 0.2 * 0.301 ~= 0.0602 / 2 ~= 0.0301
                [s],  # doc 2: 0.2 * 0.301 ~= 0.0602 / 2 ~= 0.0301
                [],   # doc 3: 0
            ],
            [  # ctx word 2: (0.0301 + 0 + 0) / 3 ~= 0.1
                [s],  # doc 1:
                [],   # doc 2: 0
                [],   # doc 3: 0
            ]
        ],
        [  # term 2
            [  # ctx word 3: tfidf = 0
                [],   # doc 1: 0
                [],   # doc 2: 0
                [],   # doc 3: 0
            ],
        ],
    ]
    texts = [  # docs
        [['This', 'is', 'doc', 'number', '1'], ['Sentence', 'number', '2']],
        [['This', 'is', 'doc', 'number', '2'], ['Sentence', 'number', '2']],
        [['This', 'is', 'doc', 'number', '3'], ['Sentence', 'number', '2']],
    ]

    # When
    result = list(tfidfs_pipeline(texts)(terms_locs))

    # Then
    assert [0.01505149978319906, 0] == result


def test_calc_avg_idf_for_context():
    # Given

    # term 0 -> agua
    # term 1 -> coche

    # text 0
    # El AGUA fría.
    # Esta frase vacía.
    # El AGUA caliente.

    # text 1
    # El coche.

    # text 2
    # empty

    locations_by_docs = [
        {  # doc
            'el': [  # context
                [(0, (0, 1)), (2, (0, 1))],  # term agua
                [(0, (0, 1)), (2, (0, 1))],  # term coche locs duplicated
            ],
            'fría': [  # context
                [(0, (2, 3))],  # term
                [],             # term
            ],
            'caliente': [  # context
                [(2, (2, 3))],  # term
                [],             # term
            ]
        },
        {  # doc
            'el': [  # context
                [(0, (0, 1))],  # term
                [(0, (0, 1))],  # term
            ]
        },
        {  # doc without matches
        }
    ]

    # When
    mean_idfs = calc_mean_doc_context_idfs(locations_by_docs)

    # Then
    assert [
               0.3767779228316687,   # mean idf for doc 0
               0.17609125905568124,  # mean idf for doc 1
               0,                    # mean idf for doc 2
           ] == mean_idfs


