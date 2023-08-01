import re

import pandas as pd
from nltk.corpus import cess_esp


def sanitize(words) -> str:
    text = ' '.join([word.replace('_', ' ') for word in words])
    text = re.sub(r'-[Ff]pa-', '(', text)
    text = re.sub(r'-[Ff]pt-', ')', text)
    text = re.sub(r'\*0\*', '', text)
    text = re.sub(r'-[Ff]e-', '', text)
    text = re.sub(r' , ', ', ', text)
    text = re.sub(r' \. ', '. ', text)
    text = re.sub(r' - - ', ' ', text)
    text = re.sub(r' \? ', '? ', text)
    text = re.sub(r' ¿ ', ' ¿', text)
    text = re.sub(r' ! ', '! ', text)
    text = re.sub(r' \( ', ' (', text)
    text = re.sub(r' \) ', ') ', text)
    text = re.sub(r' \).', ').', text)
    text = re.sub(r' \),', '),', text)
    return text


def save_tbf_to_file(fileid):
    words = cess_esp.words(fileid)
    text = sanitize(words)
    file_name = "".join(fileid.split(".")[:-1]) + ".txt"
    with open(f'data/native/{file_name}', 'w') as f:
        f.write(text)
    return f"data/native/{file_name}"


def download_native_corpus(args):
    df = pd.DataFrame({
        "Level": "Native",
        "Treebank file": cess_esp.fileids()
    })

    df["Text file"] = df["Treebank file"].apply(save_tbf_to_file)
    df = df.drop(columns=["Treebank file"])
    df.to_csv('data/native/native.csv', index=False, sep=';')
    print('Download finished successfully!')
