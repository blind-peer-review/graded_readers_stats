##############################################################################
#            LOGISTIC REGRESSION WITH WORD2VEC DOCUMENT EMBEDDINGS
##############################################################################
import os.path
import pickle

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk import flatten
from sklearn.linear_model import LogisticRegression

from graded_readers_stats.preprocess import make_stanza_docs_simple, get_fields


def document_embedding(text, w2v_model):
    """
    Converts a text document into a vector by averaging the embeddings
    of the words it contains. (Here we use a simple whitespace split.)
    """
    tokens = text.split()  # Replace with a better tokenizer if needed
    vecs = []
    for token in tokens:
        if token in w2v_model:
            vecs.append(w2v_model[token])
    if not vecs:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vecs, axis=0)


def logit_word2vec(X, y):
    """
    Trains Logistic Regression using Word2Vec document embeddings and evaluates the model.

    Parameters
    ----------
    X : pandas.Series
        Each string represents a document.

    y : pandas.Series
        Each string represents a label.
        Example labels:
            Inicial
            Intermedio
            Avanzado
    """
    global vocab
    vocab = set(' '.join(list(flatten(X.tolist()))).split())

    w2v_model = KeyedVectors.load_word2vec_format('./data/word2vec/SBW-vectors-300-min5.bin', binary=True)
    map_w2v_word_to_lemma(w2v_model)

    # Compute document embeddings as the (unweighted) average of word embeddings.
    X_embeddings = np.array([document_embedding(doc, w2v_model) for doc in X])

    # Train Logistic Regression on the document embeddings.
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_embeddings, y)

    scores = get_sorted_scores(lr, w2v_model)
    df = pd.DataFrame(scores)
    df = df[sorted(df.columns)]
    for column in df.columns:
        df[f"{column}_word"] = df[column].apply(lambda x: x[0])
        df[f"{column}_coef"] = df[column].apply(lambda x: x[1])
    return df


def map_w2v_word_to_lemma(w2v_model):
    global w2v_map
    if os.path.exists('./data/cache/w2v_lemmas.pickle'):
        with open('./data/cache/w2v_lemmas.pickle', 'rb') as file:
            w2v_map = pickle.load(file)
    else:
        w2v_st_lemmas = get_fields(make_stanza_docs_simple(w2v_model.index_to_key), 'lemma')
        w2v_lemmas = list(map(lambda x: ' '.join(flatten(x)).lower(), w2v_st_lemmas))
        w2v_map = dict(zip(w2v_model.index_to_key, w2v_lemmas))
        with open('./data/cache/w2v_lemmas.pickle', 'wb') as file:
            pickle.dump(w2v_map, file, protocol=pickle.HIGHEST_PROTOCOL)


def get_sorted_scores(model, w2v_model):
    result = {}
    cls_name_map = {
        "Inicial": "graded_1",
        "Intermedio": "graded_2",
        "Avanzado": "graded_3",
        "Infantil": "litera_1",
        "Juvenil": "litera_2",
        "Adulta": "litera_3",
    }
    coefs_per_class = model.coef_
    for cls_name, cls_idx in zip(model.classes_, range(len(model.classes_))):
        scores = []
        w_c = coefs_per_class[cls_idx]
        for word in w2v_model.index_to_key:
            vec = w2v_model[word]
            score = np.dot(w_c, vec)
            scores.append((score, w2v_map[word]))
        sorted_scores = sorted(scores, key=lambda x: abs(x[0]), reverse=True)
        filtered_scores = filter_scores(sorted_scores, vocab)
        name = cls_name_map.get(cls_name, cls_name)
        result[f"{name}"] = [(word, coef) for coef, word in filtered_scores]
    return result


def filter_scores(scores, vocab):
    seen = set()
    result = []
    for score in scores:
        word = score[1]
        if word in seen:
            continue
        seen.add(word)
        if word in vocab:
            result.append(score)
    return result


def filter_tuples(tuples_list, words_list, positive, n):
    seen = set()
    filtered_tuples = []
    for tup in tuples_list:
        coef = tup[0]
        if (positive and coef < 0) or (not positive and coef > 0):
            continue
        word = tup[1]
        if word not in seen and word in words_list:
            seen.add(word)
            filtered_tuples.append(tup)
            if len(filtered_tuples) >= n:
                break
    return filtered_tuples
