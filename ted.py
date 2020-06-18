import re
import math
import pandas as pd
from collections import Counter
from typing import List, Sequence

from models.ngrams import NgramModel

from models.skipgrams import (
	skipgram_counts_for_lines,
	apply_mask_to_ngram,
	mask_for_skipgram,
	entries_matching_skipgram
)
from utilities.dictionary import (
	sum_counters,
	keywise_quotients,
	normalize
)
from utilities.command_line import *
from utilities import librarian

### LINE OPERATIONS ###

def get_laugh_lines(lines: List[str]):
	return [start for start, end in zip(lines[:-1], lines[1:]) if end == '<Laughter>']

def get_applause_lines(lines: List[str]):
	return [start for start, end in zip(lines[:-1], lines[1:]) if end == '<Applause>']

def lines_for_tag(df, tag):
	return df.loc[tag in df['tags']]

def flatten_list(list_of_lists):
	return [item for sublist in list_of_lists for item in sublist]

##############################################

### PROCEDURES ###

def laugh_rate_analysis(df, n, count_threshold, num_to_print):

	# find ngrams in all lines
	df['all_ngrams'] = df['lines'].apply(lambda x: ngram_counts_for_lines(x, n))

	# find ngrams in laugh lines
	df['laugh_ngrams'] = df['laugh_lines'].apply(lambda x: ngram_counts_for_lines(x, n))

	laugh_rates = condition_rates_by_ngram( condition=df['laugh_ngrams'],
											overall=df['all_ngrams'],
											count_threshold=count_threshold)

	print('\nTOP LAUGH RATES (min {} occurrences):'.format(count_threshold))
	for word, quotient in top_n(laugh_rates, num_to_print):
		print(word + '\t' + str(quotient))

	print('\nLOWEST NONZERO LAUGH RATES (min {} occurrences):'.format(count_threshold))
	for word, quotient in bottom_n(laugh_rates, num_to_print):
		print(word + '\t' + str(quotient))


def show_slot_entropies(line, skipgram_entropies, skipgram_tree):
	words = line.split()
	ngrams = ngrams_for_line(line, 3)
	for ngram in ngrams:
		skipgram = apply_mask_to_ngram([True, False, True], ngram)
		print(skipgram, skipgram_entropies[skipgram], [x[0].split()[1] for x in top_n(skipgram_tree[skipgram], 10)])

def tag_analysis(df):
	tags = get_tags('tag_counts.txt')
	chosen_tag = user_choose_from_list(tags[:20])
	line_lists = [lines for i, lines in enumerate(df['lines']) if chosen_tag in df['tags'][i]]
	lines_for_tag = []
	for line_list in line_lists:
		lines_for_tag.extend(line_lists)

	print(len(lines_for_tag))

def surprise_analysis(ng_model, n, min_count_threshold, min_doc_freq_threshold):
	by_unigram_surprise, by_bigram_surprise = ng_model.ngrams_by_unigram_and_bigram_surprise(n, min_count_threshold, min_doc_freq_threshold)
	print("\n{}-GRAMS BY UNIGRAM SURPRISE".format(n))
	print_bottom_n(by_unigram_surprise, 100)
	print("\nTOP {}-GRAMS BY BIGRAM SURPRISE".format(n))
	print_top_n(by_bigram_surprise, 100)
	print("\nBOTTOM {}-GRAMS BY BIGRAM SURPRISE".format(n))
	print_bottom_n(by_bigram_surprise, 100)

def collocates_analysis(ng_model):
	print('collocates...')
	mask = user_create_mask()

	collocates = ng_model.build_collocates_from_mask(mask=mask, min_count_threshold=100)

	print('normalizing...')
	normalized_collocates = {k: normalize(v) for k, v in collocates.items()}
	print('comparing to baseline...')
	relative_collocates = {k: keywise_quotients(v, ng_model.unigram_model) for k, v in normalized_collocates.items()}
	explore_nested_dict(relative_collocates, 25)


if __name__ == '__main__':
	df = librarian.load_dataframe(cutoff=250)	# Optional: clip dataframe for testing

	# FIND LINES PRECEDING LAUGHTER AND APPLAUSE
	df['laugh_lines'] = df['lines'].apply(lambda x: get_laugh_lines(x))
	df['applause_lines'] = df['lines'].apply(lambda x: get_applause_lines(x))

	# SURPRISE MEASURES	
	lines_by_doc = df['lines']
	ng_model = NgramModel(lines_by_doc)
	#surprise_analysis(ng_model, n=6, min_count_threshold=3, min_doc_freq_threshold=5)
	collocates_analysis(ng_model)




	

