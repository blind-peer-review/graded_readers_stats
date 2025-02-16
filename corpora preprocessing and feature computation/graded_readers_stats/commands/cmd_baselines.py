import time

import numpy as np
import pandas as pd
from codetiming import Timer
from pandas.core.common import flatten
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, classification_report

from graded_readers_stats import utils
from graded_readers_stats.constants import (
    COL_LEMMA,
    COL_STANZA_DOC,
    COL_LEVEL,
)
from graded_readers_stats.data import read_pandas_csv
# from graded_readers_stats import logit
from graded_readers_stats.preprocess import (
    run,
    text_analysis_pipeline_ner,
    shrink_content_step,
)


def my_train_test_split(df, indices):
    return df.iloc[~df.index.isin(indices)], df.iloc[indices]


def confusion_matrix(y_train_true, y_train_pred,
                     y_test_true, y_test_pred,
                     column):
    print("---")
    print(f"Train data confusion matrix using '{column}' column")
    print("Matrix format:")
    print("[[TN FP]")
    print(" [FN TP]]")
    print("1. Inicial")
    print("2. Intermedio")
    print("3. Avanzado")
    print("---")
    mx_train = metrics.multilabel_confusion_matrix(y_train_true, y_train_pred[column], labels=["Inicial", "Intermedio", "Avanzado"])
    print(mx_train)
    print("---")
    print(f"Test data confusion matrix using '{column}' column")
    print("Matrix format:")
    print("[[TN FP]")
    print(" [FN TP]]")
    print("1. Inicial")
    print("2. Intermedio")
    print("3. Avanzado")
    print("---")
    mx_test = metrics.multilabel_confusion_matrix(y_test_true, y_test_pred[column], labels=["Inicial", "Intermedio", "Avanzado"])
    print(mx_test)
    # return f'{mx_train}\n---\n{mx_test}'


def execute(args):
    corpus_path = args.corpus_path

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
        # we are using the same random value from R.
        # This is because we want to synchronize the seed between R and Python.
        test_indices_readers = [1, 2, 6, 7, 9, 21, 23, 26, 29, 30, 32, 33, 34, 36, 41, 42, 46]
        test_indices_literature = []
        test_indices = test_indices_readers

        y_train_df, y_test_df = my_train_test_split(
            texts_df, test_indices
        )

        # Create an empty data frame
        y_train_pred_df = pd.DataFrame(y_train_df["Title"])
        y_test_pred_df = pd.DataFrame(y_test_df["Title"])

        # Most frequent class prediction
        y_train_pred_df["most_frequent"] = y_train_df["Level"].mode()[0]
        y_test_pred_df["most_frequent"] = y_test_df["Level"].mode()[0]

        # Weighted guessing
        probabilities = texts_df["Level"].value_counts(normalize=True)
        np.random.seed(42)
        y_train_pred_df["weighted"] = np.random.choice(
            probabilities.index.tolist(),
            size=len(y_train_pred_df),
            p=probabilities
        )
        y_test_pred_df["weighted"] = np.random.choice(
            probabilities.index.tolist(),
            size=len(y_test_pred_df),
            p=probabilities
        )

        # 1000-fold randomization
        y_train_acc = 0
        y_test_acc = 0
        acc_count = 1000
        for index in range(acc_count):
            y_train_random_pred = np.random.choice(
                probabilities.index.tolist(),
                size=len(y_train_pred_df)
            )
            y_train_acc += classification_report(
                y_train_df["Level"],
                y_train_random_pred,
                output_dict=True,
                zero_division=0
            )["accuracy"]

            y_test_random_pred = np.random.choice(
                probabilities.index.tolist(),
                size=len(y_test_pred_df)
            )
            y_test_acc += classification_report(
                y_test_df["Level"],
                y_test_random_pred,
                output_dict=True,
                zero_division=0
            )["accuracy"]
        y_train_acc_mean = y_train_acc / acc_count
        y_test_acc_mean = y_test_acc / acc_count
        print("---")
        print(f"Train data 1000-fold mean accuracy: {y_train_acc_mean}")
        print("---")
        print(f"Test data 1000-fold mean accuracy: {y_test_acc_mean}")

    # output = evaluation_and_conf_mx(
        #     y_train_df["Level"], y_train_pred_df,
        #     y_test_df["Level"], y_test_pred_df,
        #     columns=["most_frequent", "weighted"]
        # )
        # print('\n'.join(output))

        for column in ["weighted", "most_frequent"]:
            print("---")
            print(f"Train data evaluation using '{column}' column")
            print(
                classification_report(
                    y_train_df["Level"],
                    y_train_pred_df[column],
                    zero_division=0
                )
            )
            print("---")
            print(f"Test data evaluation using '{column}' column")
            print(
                classification_report(
                    y_test_df["Level"],
                    y_test_pred_df[column],
                    zero_division=0
                )
            )
            # print(
            confusion_matrix(
                y_train_df["Level"], y_train_pred_df,  # train data
                y_test_df["Level"], y_test_pred_df,    # test data
                column
            )
            # )

    with Timer(name='Export CSV', text=timer_text):
        print("")
        # file_name = corpus_path.split("/")[-1]
        # df.to_csv(f'./output/logit-bow-{file_name}', index=False)
##############################################################################
#                                   Done
##############################################################################

    print()
    utils.duration(start_main, 'Total time')
    print('')
    print('BASELINES END')
