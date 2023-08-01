env/bin/python main.py bag-of-words --strip-named-entities "data/readers/readers.csv" | tee output/readers_bow_ner.log
env/bin/python main.py bag-of-words --strip-named-entities "data/literature/literature.csv" | tee output/literature_bow_ner.log
