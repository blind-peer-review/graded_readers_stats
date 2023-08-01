env/bin/python main.py analyze "data/vocabulary.csv" "data/readers/readers.csv" Inicial | tee output/readers_1.log &
env/bin/python main.py analyze "data/vocabulary.csv" "data/readers/readers.csv" Intermedio | tee output/readers_2.log &
env/bin/python main.py analyze "data/vocabulary.csv" "data/readers/readers.csv" Avanzado | tee output/readers_3.log &
env/bin/python main.py analyze "data/vocabulary.csv" "data/literature/literature.csv" Infantil | tee output/literature_1.log &
env/bin/python main.py analyze "data/vocabulary.csv" "data/literature/literature.csv" Juvenil | tee output/literature_2.log &
env/bin/python main.py analyze "data/vocabulary.csv" "data/literature/literature.csv" Adulta | tee output/literature_3.log &
env/bin/python main.py analyze "data/vocabulary.csv" "data/native/native.csv" Native | tee output/native.log &
env/bin/python main.py analyze-documents "data/vocabulary.csv" "data/readers/readers.csv"
env/bin/python main.py analyze-documents "data/vocabulary.csv" "data/literature/literature.csv"
env/bin/python main.py analyze-documents "data/vocabulary.csv" "data/native/native.csv"
env/bin/python main.py terms_by_group "data/vocabulary.csv" --corpus_path="data/readers/readers.csv" --corpus_path="data/literature/literature.csv" --corpus_path="data/native/native.csv" | tee output/terms_by_group.log &
wait
