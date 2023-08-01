###############################################################################
# Goal: Check the overlap between BOW output and vocabulary terms.
###############################################################################
import os
from dataclasses import dataclass
from functools import reduce
from os import path

import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from ast import literal_eval
from graded_readers_stats.data import read_pandas_csv


def print_overlap(
        header: str,
        words_vocab: set[str],
        words_bow: set[str]
) -> None:
    print("---")
    print(header)
    words_overlap = words_vocab.intersection(words_bow)
    print("Overlap count: ", len(words_overlap))
    print(sorted(words_overlap))


###############################################################################
# DATA
###############################################################################

def df_for_csv(path_to_csv: str) -> pd.DataFrame:
    """
    Expected column names: name_1, name_2 i.e. <name>_<single-digit>.
    Expected row values: ('word', 0.00456) i.e. a string of tuple.
    This tuple is then converted to a real tuple using `literal_eval`.
    """
    df = read_pandas_csv(path_to_csv, sep=",")
    prefix = df.columns[0][:-1]  # get the part before the digit: name_
    for column_i in range(1, len(df.columns) + 1):
        column = f"{prefix}{column_i}"
        df[column] = df[column].apply(lambda x: literal_eval(str(x)))
    return df


readers_df = df_for_csv("../../output/logit-bow-readers.csv")
literature_df = df_for_csv("../../output/logit-bow-literature.csv")

vocabulary_csv_path = "../../data/vocabulary.csv"
vocab_df = read_pandas_csv(vocabulary_csv_path)
vocab_df["Lexical item"] = vocab_df["Lexical item"].apply(
    lambda x: x.lower().strip()
)

vocab_all_words = set(vocab_df["Lexical item"])
vocab_level1_words = set(
    vocab_df[vocab_df["Level"] == "A1-A2"]["Lexical item"])
vocab_level2_words = set(vocab_df[vocab_df["Level"] == "B1"]["Lexical item"])
vocab_level3_words = set(vocab_df[vocab_df["Level"] == "B2"]["Lexical item"])

# Graded
bow_graded_1_all = {name: value for name, value in readers_df["graded_1"]}
bow_graded_1_pos = {k: v for k, v in bow_graded_1_all.items() if v > 0}
bow_graded_1_neg = {k: abs(v) for k, v in bow_graded_1_all.items() if v < 0}
bow_graded_2_all = {name: value for name, value in readers_df["graded_2"]}
bow_graded_2_pos = {k: v for k, v in bow_graded_2_all.items() if v > 0}
bow_graded_2_neg = {k: abs(v) for k, v in bow_graded_2_all.items() if v < 0}
bow_graded_3_all = {name: value for name, value in readers_df["graded_3"]}
bow_graded_3_pos = {k: v for k, v in bow_graded_3_all.items() if v > 0}
bow_graded_3_neg = {k: abs(v) for k, v in bow_graded_3_all.items() if v < 0}
bow_graded_all = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_all, bow_graded_2_all, bow_graded_3_all)
)
bow_graded_pos = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_pos, bow_graded_2_pos, bow_graded_3_pos)
)
bow_graded_neg = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_neg, bow_graded_2_neg, bow_graded_3_neg)
)

# Literature
bow_litera_1_all = {name: value for name, value in literature_df["litera_1"]}
bow_litera_1_pos = {k: v for k, v in bow_litera_1_all.items() if v > 0}
bow_litera_1_neg = {k: abs(v) for k, v in bow_litera_1_all.items() if v < 0}
bow_litera_2_all = {name: value for name, value in literature_df["litera_2"]}
bow_litera_2_pos = {k: v for k, v in bow_litera_2_all.items() if v > 0}
bow_litera_2_neg = {k: abs(v) for k, v in bow_litera_2_all.items() if v < 0}
bow_litera_3_all = {name: value for name, value in literature_df["litera_3"]}
bow_litera_3_pos = {k: v for k, v in bow_litera_3_all.items() if v > 0}
bow_litera_3_neg = {k: abs(v) for k, v in bow_litera_3_all.items() if v < 0}
bow_litera_all = reduce(
    lambda x, y: dict(x, **y),
    (bow_litera_1_all, bow_litera_2_all, bow_litera_3_all)
)
bow_litera_pos = reduce(
    lambda x, y: dict(x, **y),
    (bow_litera_1_pos, bow_litera_2_pos, bow_litera_3_pos)
)
bow_litera_neg = reduce(
    lambda x, y: dict(x, **y),
    (bow_litera_1_neg, bow_litera_2_neg, bow_litera_3_neg)
)

# Graded + Literature
bow_all = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_all, bow_graded_2_all, bow_graded_3_all,
     bow_litera_1_all, bow_litera_2_all, bow_litera_3_all)
)
bow_pos = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_pos, bow_graded_2_pos, bow_graded_3_pos,
     bow_litera_1_pos, bow_litera_2_pos, bow_litera_3_pos)
)
bow_neg = reduce(
    lambda x, y: dict(x, **y),
    (bow_graded_1_neg, bow_graded_2_neg, bow_graded_3_neg,
     bow_litera_1_neg, bow_litera_2_neg, bow_litera_3_neg)
)


##############################################################################
# WordCloud generation
##############################################################################


@dataclass
class WordCloudSection:
    mask: str
    output: str
    wordToFreqMap: dict[str: float]


def draw_wc_section(section: WordCloudSection):
    d = os.getcwd()
    mask = np.array(Image.open(path.join(d, section.mask)))
    wc = WordCloud(
        mode="RGBA",
        background_color=None,
        width=2800,
        height=2800,
        scale=1,
        mask=mask
    )
    if len(section.wordToFreqMap.items()):
        wc.generate_from_frequencies(section.wordToFreqMap)
        wc.to_file(path.join(d, section.output))


freqs_pos = {
    "to1": {
        k: v
        for k, v in bow_graded_1_pos.items()
        if k in set(bow_graded_1_pos.keys()).difference(vocab_level1_words)
    },
    "to2": {
        k: v
        for k, v in bow_graded_2_pos.items()
        if k in set(bow_graded_2_pos.keys()).difference(vocab_level2_words)
    },
    "to3": {
        k: v
        for k, v in bow_graded_3_pos.items()
        if k in set(bow_graded_3_pos.keys()).difference(vocab_level3_words)
    },
    "ti1": {
        k: v
        for k, v in bow_graded_1_pos.items()
        if k in set(bow_graded_1_pos.keys()).intersection(vocab_level1_words)
    },
    "ti2": {
        k: v
        for k, v in bow_graded_2_pos.items()
        if k in set(bow_graded_2_pos.keys()).intersection(vocab_level2_words)
    },
    "ti3": {
        k: v
        for k, v in bow_graded_3_pos.items()
        if k in set(bow_graded_3_pos.keys()).intersection(vocab_level3_words)
    },
    "bo1": {
        k: v
        for k, v in bow_litera_1_pos.items()
        if k in set(bow_litera_1_pos.keys()).difference(vocab_level1_words)
    },
    "bo2": {
        k: v
        for k, v in bow_litera_2_pos.items()
        if k in set(bow_litera_2_pos.keys()).difference(vocab_level2_words)
    },
    "bo3": {
        k: v
        for k, v in bow_litera_3_pos.items()
        if k in set(bow_litera_3_pos.keys()).difference(vocab_level3_words)
    },
    "bi1": {
        k: v
        for k, v in bow_litera_1_pos.items()
        if k in set(bow_litera_1_pos.keys()).intersection(vocab_level1_words)
    },
    "bi2": {
        k: v
        for k, v in bow_litera_2_pos.items()
        if k in set(bow_litera_2_pos.keys()).intersection(vocab_level2_words)
    },
    "bi3": {
        k: v
        for k, v in bow_litera_3_pos.items()
        if k in set(bow_litera_3_pos.keys()).intersection(vocab_level3_words)
    },
}
freqs_neg = {
    "to1": {
        k: v
        for k, v in bow_graded_1_neg.items()
        if k in set(bow_graded_1_neg.keys()).difference(vocab_level1_words)
    },
    "to2": {
        k: v
        for k, v in bow_graded_2_neg.items()
        if k in set(bow_graded_2_neg.keys()).difference(vocab_level2_words)
    },
    "to3": {
        k: v
        for k, v in bow_graded_3_neg.items()
        if k in set(bow_graded_3_neg.keys()).difference(vocab_level3_words)
    },
    "ti1": {
        k: v
        for k, v in bow_graded_1_neg.items()
        if k in set(bow_graded_1_neg.keys()).intersection(vocab_level1_words)
    },
    "ti2": {
        k: v
        for k, v in bow_graded_2_neg.items()
        if k in set(bow_graded_2_neg.keys()).intersection(vocab_level2_words)
    },
    "ti3": {
        k: v
        for k, v in bow_graded_3_neg.items()
        if k in set(bow_graded_3_neg.keys()).intersection(vocab_level3_words)
    },
    "bo1": {
        k: v
        for k, v in bow_litera_1_neg.items()
        if k in set(bow_litera_1_neg.keys()).difference(vocab_level1_words)
    },
    "bo2": {
        k: v
        for k, v in bow_litera_2_neg.items()
        if k in set(bow_litera_2_neg.keys()).difference(vocab_level2_words)
    },
    "bo3": {
        k: v
        for k, v in bow_litera_3_neg.items()
        if k in set(bow_litera_3_neg.keys()).difference(vocab_level3_words)
    },
    "bi1": {
        k: v
        for k, v in bow_litera_1_neg.items()
        if k in set(bow_litera_1_neg.keys()).intersection(vocab_level1_words)
    },
    "bi2": {
        k: v
        for k, v in bow_litera_2_neg.items()
        if k in set(bow_litera_2_neg.keys()).intersection(vocab_level2_words)
    },
    "bi3": {
        k: v
        for k, v in bow_litera_3_neg.items()
        if k in set(bow_litera_3_neg.keys()).intersection(vocab_level3_words)
    },
}

for name, freqs in [("pos", freqs_pos), ("neg", freqs_neg)]:
    for index in [1, 2, 3]:
        for section in ["to", "ti", "bo", "bi"]:
            sectionName = f"{section}{index}"
            draw_wc_section(
                WordCloudSection(
                    mask=f"masks/{sectionName}.jpg",
                    output=f"output/{name}/{sectionName}.png",
                    wordToFreqMap=freqs[sectionName]
                )
            )


# Command line join background image with generated overlay words
# magick convert -page 2800x2800+0+0 bg.jpg -page +0+0 bi1.png -page +0+0 bi2.png -page +0+0 bi3.png -page +0+0 bo1.png -page +0+0 bo2.png -page +0+0 bo3.png -page +0+0 ti1.png -page +0+0 ti2.png -page +0+0 ti3.png -page +0+0 to1.png -page +0+0 to2.png -page +0+0 to3.png -background white -flatten output.jpg

print("Done!")
