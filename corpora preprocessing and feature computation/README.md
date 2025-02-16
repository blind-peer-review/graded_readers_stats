# Corpora pre-processing and feature computation

Here is the code written to:

* Perform tokenization, multi-word token expansion, lemmatization, part-of-speech and morphological feature tagging, dependency parsing, and named entity recognition on the graded readers, the literary works, the reference corpus, and the vocabulary data set.

* Trace and record every occurrence of graded vocabulary in the corpora.

* Compute the 40 features chosen to conduct this study and obtain the file used later on to perform relevant statistical analyses (refer to the 'linguistic complexity assessment' folder).

Please note that access to the complete data cannot be granted due to copyright restrictions, so dummy text files to run the code are provided instead.

## Requirements

Python version 3.10

## Installation

Execute in terminal at the root of the project folder

```shell
# create a virtual environment
python -m venv env
source env/bin/activate

# download code dependencies
pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# download supporting files
python -c 'import stanza; stanza.download("es")'
python -c 'import nltk; nltk.download("cess_esp"); nltk.download("names"); nltk.download("stopwords")'
```

## Usage

```shell
./scripts/main_final.sh
```
