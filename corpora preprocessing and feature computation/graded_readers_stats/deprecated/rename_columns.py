import pandas as pd

df1 = pd.read_csv(
    'graded_readers_stats.csv'
)
# for col in df1.columns:
#     print(col)

df2 = pd.read_csv(
    '../../output/main.csv'
)
# for col in df2.columns:
#     print(col)

# No need to touch
# "Lexical item": "Lexical item",
# "Level": "Level",
# "Topic": "Topic",
# "Subtopic": "Subtopic",
# "Lemma": "Lemma",

# not present in new files
# drop_columns = {
#     "Pre-processing",
#     "Occurrence locations (Graded readers)",
#     "Occurrence locations (Literature)",
#     "Occurrence locations (Native corpus)",
#     "Contexts of occurrence (Graded readers)",
#     "Contexts of occurrence (Literature)",
#     "Contexts of occurrence (Native corpus)",
#     "Contexts of occurrence locations (Graded readers)",
#     "Contexts of occurrence locations (Literature)",
#     "Contexts of occurrence locations (Native corpus)",
#     "Trees and locations (Inicial)",
#     "Trees and locations (Intermedio)",
#     "Trees and locations (Avanzado)",
#     "Trees and locations (Infantil)",
#     "Trees and locations (Juvenil)",
#     "Trees and locations (Adulta)",
#     "Trees and locations (Native corpus)",
# }

new_to_old_mapping = {
    # _f(yi(_aCount_<Esc>pldt"_
    # Count
    "Count_Inicial": "Occurrence count (Inicial)",
    "Count_Intermedio": "Occurrence count (Intermedio)",
    "Count_Avanzado": "Occurrence count (Avanzado)",
    "Count_Infantil": "Occurrence count (Infantil)",
    "Count_Juvenil": "Occurrence count (Juvenil)",
    "Count_Adulta": "Occurrence count (Adulta)",
    "Count_Native": "Occurrence count (Native corpus)",

    # _f(yi(_aTotal_<Esc>pldt"_
    # Total
    "Total_Inicial": "Total word count (Inicial)",
    "Total_Intermedio": "Total word count (Intermedio)",
    "Total_Avanzado": "Total word count (Avanzado)",
    "Total_Infantil": "Total word count (Infantil)",
    "Total_Juvenil": "Total word count (Juvenil)",
    "Total_Adulta": "Total word count (Adulta)",
    "Total_Native": "Total word count (Native corpus)",

    # _f(yi(_aFrequency_<Esc>pldt"_
    # Frequency
    "Frequency_Inicial": "Frequency of occurrence (Inicial)",
    "Frequency_Intermedio": "Frequency of occurrence (Intermedio)",
    "Frequency_Avanzado": "Frequency of occurrence (Avanzado)",
    "Frequency_Infantil": "Frequency of occurrence (Infantil)",
    "Frequency_Juvenil": "Frequency of occurrence (Juvenil)",
    "Frequency_Adulta": "Frequency of occurrence (Adulta)",
    "Frequency_Native": "Frequency of occurrence (Native corpus)",

    # _f(yi(_aContext_count_<Esc>pldt"j_
    # Context_count
    "Context_count_Inicial": "Contexts of occurrence count (Inicial)",
    "Context_count_Intermedio": "Contexts of occurrence count (Intermedio)",
    "Context_count_Avanzado": "Contexts of occurrence count (Avanzado)",
    "Context_count_Infantil": "Contexts of occurrence count (Infantil)",
    "Context_count_Juvenil": "Contexts of occurrence count (Juvenil)",
    "Context_count_Adulta": "Contexts of occurrence count (Adulta)",
    "Context_count_Native": "Contexts of occurrence count (Native corpus)",

    # _f(yi(_aContext_frequency_<Esc>pldt"j_
    # Context_frequency
    "Context_frequency_Inicial": "Frequency of contexts of occurrence (Inicial)",
    "Context_frequency_Intermedio": "Frequency of contexts of occurrence (Intermedio)",
    "Context_frequency_Avanzado": "Frequency of contexts of occurrence (Avanzado)",
    "Context_frequency_Infantil": "Frequency of contexts of occurrence (Infantil)",
    "Context_frequency_Juvenil": "Frequency of contexts of occurrence (Juvenil)",
    "Context_frequency_Adulta": "Frequency of contexts of occurrence (Adulta)",
    "Context_frequency_Native": "Frequency of contexts of occurrence (Native corpus)",

    # _f(yi(_aTree_<Esc>pldt"j_
    # Tree
    "Tree_Inicial": "Trees' min, max, and average width & depth (Inicial)",
    "Tree_Intermedio": "Trees' min, max, and average width & depth (Intermedio)",
    "Tree_Avanzado": "Trees' min, max, and average width & depth (Avanzado)",
    "Tree_Infantil": "Trees' min, max, and average width & depth (Infantil)",
    "Tree_Juvenil": "Trees' min, max, and average width & depth (Juvenil)",
    "Tree_Adulta": "Trees' min, max, and average width & depth (Adulta)",
    "Tree_Native": "Trees' min, max, and average width & depth (Native corpus)",

    # _f(yi(_aTFIDF_<Esc>pldt"j_
    # TFIDF
    "TFIDF_Inicial": "TFIDF (Inicial)",
    "TFIDF_Intermedio": "TFIDF (Intermedio)",
    "TFIDF_Avanzado": "TFIDF (Avanzado)",
    "TFIDF_Infantil": "TFIDF (Infantil)",
    "TFIDF_Juvenil": "TFIDF (Juvenil)",
    "TFIDF_Adulta": "TFIDF (Adulta)",
    "TFIDF_Native": "TFIDF (Native corpus)",
}

new_scheme_columns = {
    "Lexical item",
    "Level",
    "Topic",
    "Subtopic",
    "Lemma",

    "Count_Inicial",
    "Total_Inicial",
    "Frequency_Inicial",
    "TFIDF_Inicial",
    "Tree_Inicial",

    "Context_count_Inicial",
    "Context_total_Inicial",
    "Context_frequency_Inicial",
    "Context_TFIDF_Inicial",
    "Context_tree_Inicial",

    # Pattern repeats for every level
    # Note that "Native corpus" is just "Native" now
}
