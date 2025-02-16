import argparse

from graded_readers_stats import _torch_patch
from graded_readers_stats.commands import (
    cmd_analyze_documents,
    cmd_analyze_vocabulary,
    cmd_bag_of_words,
    cmd_download_native_corpus,
    cmd_merge_output,
    cmd_stats_for_texts,
    cmd_terms_by_group,
    cmd_stats_for_terms,
    cmd_baselines,
)

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

##############################################################################
#                          Download Native Corpus
##############################################################################

subparser = subparsers.add_parser(
    'download-native-corpus',
    help='Download cess_esp file from NLTK and generate text and csv files')
subparser.set_defaults(
    func=cmd_download_native_corpus.download_native_corpus
)

##############################################################################
#                              Terms by group                                #
##############################################################################

subparser = subparsers.add_parser(
    'terms_by_group',
    help='Determine presence of a term in groups (e.g. Inicial, Intermedio).')
subparser.add_argument(
    'vocabulary_path',
    help='file path to a CSV with terms/vocabulary')
subparser.add_argument(
    '--corpus_path',
    action="append",
    help='file path to a CSV with file paths to texts')
subparser.add_argument(
    '--max_terms', type=int,
    help='max number of terms to analyze. Useful for tests')
subparser.add_argument(
    '--max_docs', type=int,
    help='max number of docs to analyze. Useful for tests')
subparser.set_defaults(func=cmd_terms_by_group.analyze)

##############################################################################
#                            Analyze Vocabulary                              #
##############################################################################

subparser = subparsers.add_parser(
    'analyze',
    help='Calculate Freq, TFIDF and Tree per vocabulary item.')
subparser.add_argument(
    'vocabulary_path',
    help='file path to a CSV with terms/vocabulary')
subparser.add_argument(
    'corpus_path',
    help='file path to a CSV with file paths to texts')
subparser.add_argument(
    'level',
    help='only process texts of the specified level '
         'column e.g. --level=Inicial would process '
         'only rows with a value Inicial for the column '
         'Level that is assumed to be present in all '
         'loaded texts.')
subparser.add_argument(
    '--max_terms', type=int,
    help='max number of terms to analyze. Useful for tests')
subparser.add_argument(
    '--max_docs', type=int,
    help='max number of docs to analyze. Useful for tests')
subparser.set_defaults(func=cmd_analyze_vocabulary.analyze)

##############################################################################
#                            Analyze Documents                               #
##############################################################################

subparser = subparsers.add_parser(
    'analyze-documents',
    help='Calculate Freq, TFIDF and Tree per document')
subparser.add_argument(
    'vocabulary_path',
    help='file path to a CSV with terms/vocabulary')
subparser.add_argument(
    'corpus_path',
    help='file path to a CSV with file paths to texts')
subparser.add_argument(
    '--max_terms', type=int,
    help='max number of terms to analyze. Useful for tests')
subparser.add_argument(
    '--max_docs', type=int,
    help='max number of docs to analyze. Useful for tests')
subparser.set_defaults(func=cmd_analyze_documents.analyze)


##############################################################################
#                              Stats for terms                               #
##############################################################################

subparser = subparsers.add_parser(
    'stats_for_terms',
    help='Stats based on stanza')
subparser.add_argument(
    '--vocabulary_path',
    help='file path(s) to a CSV with file paths to texts')
subparser.add_argument(
    '--max_docs', type=int,
    help='max number of docs to analyze. Useful for tests')
subparser.set_defaults(func=cmd_stats_for_terms.execute)

##############################################################################
#                              Stats for texts                               #
##############################################################################

subparser = subparsers.add_parser(
    'stats_for_texts',
    help='Stats based on stanza')
subparser.add_argument(
    '--corpus_paths',
    nargs='+',
    help='file path(s) to a CSV with file paths to texts')
subparser.add_argument(
    '--max_docs', type=int,
    help='max number of docs to analyze. Useful for tests')
subparser.set_defaults(func=cmd_stats_for_texts.execute)

##############################################################################
#                               Bag of Words                                 #
##############################################################################

subparser = subparsers.add_parser(
    'bag-of-words',
    help='Predict level using BoW')
subparser.add_argument(
    'corpus_path',
    help='file path to a CSV with file paths to texts')
subparser.add_argument(
    '--max_docs', type=int,
    help='max number of docs to analyze. Useful for tests')
subparser.add_argument(
    '--shorten-content', action="store_true",
    help='shortens content of docs for faster processing.')
subparser.add_argument(
    '--strip-named-entities', action='store_true',
    help='detect and strip Named Entities using Stanza.')
subparser.set_defaults(func=cmd_bag_of_words.execute)

##############################################################################
#                                Baselines                                   #
##############################################################################

subparser = subparsers.add_parser(
    'baselines',
    help='calculate baselines')
subparser.add_argument(
    'corpus_path',
    help='file path to a CSV with file paths to texts')
subparser.set_defaults(func=cmd_baselines.execute)

##############################################################################
#                               Merge output
##############################################################################

subparser = subparsers.add_parser(
    'merge-output',
    help='Merge all CSV files into a single main.csv')
subparser.set_defaults(func=cmd_merge_output.merge_output)

##############################################################################
#                                  MAIN
##############################################################################

print()
print('---')
print()
print('Parsing arguments...')
args = parser.parse_args()

print('Parsing done:', args)
args.func(args)
