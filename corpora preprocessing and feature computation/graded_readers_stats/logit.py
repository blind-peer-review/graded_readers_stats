##############################################################################
#                          LOGISTIC REGRESSION
##############################################################################
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.model_selection import train_test_split, cross_val_score


def logit(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y, test_size=0.3
    )
    n_classes = np.unique(y).shape[0]  # number of graded levels in our case
    X_train_index = X_train.index
    X_test_index = X_test.index
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    tfidf = TfidfTransformer()
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.fit_transform(X_test)
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=100,
        random_state=42,
    )
    lr.fit(X_train, y_train)

    print('=== BOW ===')

    # IMPORTANT:
    # Cross-validation was used with the full dataset.
    # However, it is disabled when using the small, demo-size dataset!

    # scores = cross_val_score(lr, X_train, y_train, cv=5)
    # print('Cross-validation scores:', scores, sep='\n')
    # print('Cross-validation avg. score:', scores.mean(), sep='\n')

    print('Model train score:', lr.score(X_train, y_train), sep='\n')
    print('Model test score:', lr.score(X_test, y_test), sep='\n')

    y_train_pred = pd.DataFrame(lr.predict(X_train), index=X_train_index,
                                columns=['lr_bow_pred'])
    y_test_pred = pd.DataFrame(lr.predict(X_test), index=X_test_index,
                               columns=['lr_bow_pred'])

    columns = ['lr_bow_pred']

    print_report(
        y_train, y_train_pred,
        y_test, y_test_pred,
        lr, vectorizer, columns
    )
    return make_dataframe(lr, vectorizer, 100)


def evaluate(y_train_true, y_train_pred, y_test_true, y_test_pred, column):
    labels = [
        'accuracy',
        'precision',
        'recall',
        'f1',
    ]
    average = "micro"
    train_values = [
        accuracy_score(y_train_true, y_train_pred[column]),
        precision_score(y_train_true, y_train_pred[column],
                        average=average, zero_division=0),
        recall_score(y_train_true, y_train_pred[column], average=average),
        f1_score(y_train_true, y_train_pred[column], average=average)]
    test_values = [
        accuracy_score(y_test_true, y_test_pred[column]),
        precision_score(y_test_true, y_test_pred[column],
                        average=average, zero_division=0),
        recall_score(y_test_true, y_test_pred[column], average=average),
        f1_score(y_test_true, y_test_pred[column], average=average)]

    output = []
    for (label, train, test) in zip(labels, train_values, test_values):
        output.append('{: >10} {:0.6f} {:0.6f}'.format(label, train, test))
    return output


def confusion_matrix(y_train_true, y_train_pred,
                     y_test_true, y_test_pred,
                     column):
    mx_train = metrics.confusion_matrix(y_train_true, y_train_pred[column])
    mx_test = metrics.confusion_matrix(y_test_true, y_test_pred[column])
    return f'{mx_train}\n---\n{mx_test}'


def evaluation_and_conf_mx(y_train_true, y_train_pred,
                           y_test_true, y_test_pred,
                           columns):
    output = []
    for column in columns:
        output.append('------------------------------')
        output.append('\t\t' + column)
        output.append('------------------------------')
        output.append('             Train  |  Test')
        output.append('------------------------------')
        output += evaluate(y_train_true, y_train_pred,
                           y_test_true, y_test_pred,
                           column)
        output.append('------------------------------')
        output += [confusion_matrix(y_train_true, y_train_pred,
                                    y_test_true, y_test_pred,
                                    column)]
        output += '\n'
    return output


def show_most_informative_features(model, vectorizer, n):
    output = []
    coefs_per_class = model.coef_
    for cls_name, cls_idx in zip(model.classes_, range(3)):
        coefs_and_names = sorted(
            zip(coefs_per_class[cls_idx], vectorizer.get_feature_names_out()),
            key=itemgetter(0), reverse=True)
        topn = list(zip(coefs_and_names[:n], coefs_and_names[:-(n + 1):-1]))

        output.append('---------------------------------------------------')
        output.append(f"                     {cls_name}")
        output.append('---------------------------------------------------')
        for (cp, fnp), (cn, fnn) in topn:
            # cp = coefficient positive
            # cn = coefficient negative
            # fnp = feature name positive
            # fnn = feature name negative
            output.append(
                "{:0.6f}{: >15}    {:0.6f}{: >15}".format(cp, fnp, cn, fnn))

        output.append('---------------------------------------------------')
        output.append(f"                 {cls_name} as dict")
        output.append('---------------------------------------------------')
        output.append(f"{cls_name} = {{")  # '{{' is an escaped '{' in f-string
        for (cp, fnp), (_, _) in topn:
            output.append(f'\t"{fnp}": {cp:.6f},')
        for (_, _), (cn, fnn) in topn:
            output.append(f'\t"{fnn}": {cn:.6f},')
        output.append("}")

    return "\n".join(output)


def print_report(
        y_train_true, y_train_pred,
        y_test_true, y_test_pred,
        model, vectorizer, columns
):
    output = evaluation_and_conf_mx(
        y_train_true, y_train_pred,
        y_test_true, y_test_pred,
        columns)
    output.append('  Coef          Feature    Coef            Feature')
    # output.append('--------------------------------------------------')
    output += [show_most_informative_features(model, vectorizer, 60)]
    print('\n'.join(output))


def make_dataframe(model, vectorizer, n):
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
    for cls_name, cls_idx in zip(model.classes_, range(3)):
        coefs_and_names = sorted(
            zip(coefs_per_class[cls_idx], vectorizer.get_feature_names_out()),
            key=itemgetter(0),
            reverse=True
        )
        topn = list(coefs_and_names[:n] + coefs_and_names[:-(n + 1):-1])
        name = cls_name_map[cls_name]
        result[f"{name}"] = [(f_name, coef) for (coef, f_name) in topn]

    df = pd.DataFrame(result)
    return df

