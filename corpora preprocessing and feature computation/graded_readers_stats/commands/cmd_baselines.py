import time

import numpy as np
import pandas as pd
from codetiming import Timer
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, classification_report, multilabel_confusion_matrix

from graded_readers_stats import utils
from graded_readers_stats.data import read_pandas_csv


def my_train_test_split(df, indices):
    return df.iloc[~df.index.isin(indices)], df.iloc[indices]


def calc_accuracy(tn, fp, fn, tp) -> float:
    if (tp + fp + tn + fn) == 0:
        return 0
    return (tp + tn) / (tp + fp + tn + fn)


def calc_balanced_accuracy(tn, fp, fn, tp):
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return (sensitivity + specificity) / 2


def calc_precision(fp, tp) -> float:
    if (tp + fp) == 0:
        return 0
    return tp / (tp + fp)


def calc_recall(fn, tp) -> float:
    if (tp + fn) == 0:
        return 0
    return tp / (tp + fn)


def calc_f1(precision, recall) -> float:
    if (precision + recall) == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def my_classification_report(y_gold, y_pred, labels, cm_name=""):
    cm = multilabel_confusion_matrix(y_gold, y_pred, labels=labels)
    report = {}
    for index, label in enumerate(labels):
        tn, fp, fn, tp = cm[index].ravel()

        if cm_name != "":
            print(f"{cm_name} confusion matrix for {label}")
            print(" tn fp")
            print(" fn tp")
            print()
            print(f"{tn:>3}{fp:>3}")
            print(f"{fn:>3}{tp:>3}")
            print()

        precision = calc_precision(fp, tp)
        recall = calc_recall(fn, tp)
        report[label] = {
            "balanced": calc_balanced_accuracy(tn, fp, fn, tp),
            "accuracy": calc_accuracy(tn, fp, fn, tp),
            "precision": precision,
            "recall": recall,
            "f1": calc_f1(precision, recall)
        }
    return report


def confusion_matrix(y_true, y_pred, name, column, labels):
    print("---")
    print(f"{name} data confusion matrix using '{column}' column")
    print("Matrix format:")
    print("[[TN FP]")
    print(" [FN TP]]")
    print(f"1. {labels[0]}")
    print(f"2. {labels[1]}")
    print(f"3. {labels[2]}")
    print("---")
    print(
        metrics.multilabel_confusion_matrix(
            y_true, y_pred[column], labels=labels
        )
    )


def execute(args):
    corpus_path = args.corpus_path
    labels = args.labels.split(",")
    test_indices = list(map(int, args.indices.split(",")))

    print()
    print('BASELINES START')
    print('---')
    print('corpus_path = ', corpus_path)
    print('---')

    timer_text = '{name}: {:0.0f} seconds'
    start_main = time.time()

    ##############################################################################
    #                                Preprocess                                  #
    ##############################################################################

    with Timer(name='Load data', text=timer_text):
        texts_df = read_pandas_csv(corpus_path)

    ##############################################################################
    #                             Baselines
    ##############################################################################

    with Timer(name='Baselines', text=timer_text):
        # Instead of using random train-to-test split,
        # we are using the random seed generated previously in R.
        test_indices_readers = [
            1, 2, 6, 7, 9, 21, 23, 26, 29, 30, 32, 33, 34, 36, 41, 42, 46
        ]
        test_indices_literature = [
            0, 3, 8, 14, 17, 21, 23, 25, 26, 31, 34, 35, 42, 45, 46
        ]
        # test_indices = test_indices_readers

        _, y_test_df = my_train_test_split(
            texts_df, test_indices
        )

        # Create an empty data frame
        y_test_pred_df = pd.DataFrame(y_test_df["Title"])

        # Most frequent class prediction
        # The mode of a set of values is the value that appears most often.
        # 0th element is the most frequent one.
        y_test_pred_df["most_frequent"] = y_test_df["Level"].mode()[0]

        # Weighted guessing
        np.random.seed(42)
        probabilities = texts_df["Level"].value_counts(normalize=True)
        y_test_pred_df["weighted"] = np.random.choice(
            probabilities.index.tolist(),
            size=len(y_test_pred_df),
            p=probabilities
        )

        # 1000-fold randomization
        random_count = 1000
        result = {}
        for label in labels:
            result[label] = {
                "balanced": 0,
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
            }

        y_gold = y_test_df["Level"]
        for index in range(random_count):
            y_rand_pred = np.random.choice(labels, size=len(y_test_df))
            report = my_classification_report(
                y_gold,
                y_rand_pred,
                labels
            )
            # accumulate scores from each iteration
            for label in labels:
                result[label]["balanced"] += report[label]["balanced"]
                result[label]["accuracy"] += report[label]["accuracy"]
                result[label]["precision"] += report[label]["precision"]
                result[label]["recall"] += report[label]["recall"]
                result[label]["f1"] += report[label]["f1"]

        # find averages
        for label in labels:
            result[label]["balanced"] /= random_count
            result[label]["accuracy"] /= random_count
            result[label]["precision"] /= random_count
            result[label]["recall"] /= random_count
            result[label]["f1"] /= random_count

        for column in ["weighted", "most_frequent"]:
            print("---")
            column_result = my_classification_report(
                y_gold,
                y_test_pred_df[column],
                labels,
                cm_name=column
            )
            for label in labels:
                print(f"{column} scores for {label}")
                for key in column_result[label]:
                    print(f"\t{key:>9} = {column_result[label][key]:.2f}")

        print("---")
        for label in labels:
            print(f"1000-fold randomization scores for {label}")
            for key in result[label]:
                print(f"\t{key:>9} = {result[label][key]:.2f}")

        # confusion_matrix(
        #         y_gold, y_test_pred_df,
        #         name="Test",
        #         column=column,
        #         labels=labels
        #     )

    ##############################################################################
    #                                   Done
    ##############################################################################

    print()
    utils.duration(start_main, 'Total time')
    print('')
    print('BASELINES END')
