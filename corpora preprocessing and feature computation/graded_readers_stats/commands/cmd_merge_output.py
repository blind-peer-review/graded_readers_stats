import ntpath
import os

import pandas as pd


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def split_tree_column(tree_props):
    def parse_float(str_float):
        try:
            return float(str_float)
        except ValueError:
            return str_float

    def parse_tuple(str_tuple) -> (int, int, int, int, int, int):
        result = str_tuple\
            .replace('(', '')\
            .replace(')', '')\
            .replace(' ', '')\
            .split(',')
        result = list(map(parse_float, result))
        return result

    tree_props = list(map(parse_tuple, tree_props))
    props_by_column = list(zip(*tree_props))
    return props_by_column


def merge_output(args):
    cwd = os.path.abspath('./output')
    files = os.listdir(cwd)
    output = pd.DataFrame()

    for file in files:
        if not file.endswith('.csv') or file.endswith('main.csv'):
            continue

        level, _ = path_leaf(file).split('.')
        df = pd.read_csv(f'./output/{file}')
        df = df.drop(columns=[
            'Context words',
            'Context count per word'
        ])
        # Avoid re-adding shared columns such as Lexical item, Topic, Subtopic
        common_columns_already_added = len(output) > 0
        if common_columns_already_added:
            df = df.drop(columns=[
                'Lexical item',
                'Level',
                'Topic',
                'Subtopic',
                'Lemma',
            ])
        # Split Tree props into columns
        # (min_width, max_width, avg_width, min_height, max_height, avg_height)
        props_by_column = split_tree_column(df['Tree'])
        df[f'Tree_{level}_MinW'] = props_by_column[0]
        df[f'Tree_{level}_MaxW'] = props_by_column[1]
        df[f'Tree_{level}_AvgW'] = props_by_column[2]
        df[f'Tree_{level}_MinH'] = props_by_column[3]
        df[f'Tree_{level}_MaxH'] = props_by_column[4]
        df[f'Tree_{level}_AvgH'] = props_by_column[5]
        # df = df.drop(columns=['Tree'])

        props_by_column = split_tree_column(df['Context tree'])
        df[f'Context_tree_{level}_MinW'] = props_by_column[0]
        df[f'Context_tree_{level}_MaxW'] = props_by_column[1]
        df[f'Context_tree_{level}_AvgW'] = props_by_column[2]
        df[f'Context_tree_{level}_MinH'] = props_by_column[3]
        df[f'Context_tree_{level}_MaxH'] = props_by_column[4]
        df[f'Context_tree_{level}_AvgH'] = props_by_column[5]
        # df = df.drop(columns=['Context tree'])

        df = df.rename(columns={
            "Count": f"Count_{level}",
            "Total": f"Total_{level}",
            "Frequency": f"Frequency_{level}",
            "TFIDF": f"TFIDF_{level}",
            "Tree": f"Tree_{level}",

            "Context count": f"Context_count_{level}",
            "Context total": f"Context_total_{level}",
            "Context frequency": f"Context_frequency_{level}",
            "Context TFIDF": f"Context_TFIDF_{level}",
            "Context tree": f"Context_tree_{level}",
        })
        output = pd.concat([output, df], axis=1)

    output.head()
    output.to_csv(f'./output/main.csv')
