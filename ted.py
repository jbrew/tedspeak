import re
import math
import pandas as pd
from collections import Counter

from models.ngrams import (
	NgramModel,
	ngram_counts_for_lines,
)

from utilities.dictionary import (
	sum_counters,
	keywise_quotients,
	normalize,
	keywise_rates_of_condition,
	top_n,
	bottom_n
)
from utilities.command_line import (
	user_create_mask,
	explore_nested_dict,
	print_top_n,
	print_bottom_n
)

##############################################

### PROCEDURES ###

def surprise_analysis(ng_model, n, min_count_threshold, min_doc_freq_threshold, num_to_print=20):
	"""
	The "unigram surprise" of an ngram is the ratio of its observed occurrence rate
	to its expected occurrence rate under a unigram (i.e., bag of words) model.

	Similarly, the "bigram surprise" is the ratio of its observed rate to its
	expected rate under a bigram model.
	"""
	by_unigram_surprise, by_bigram_surprise = ng_model.ngrams_by_unigram_and_bigram_surprise(n, min_count_threshold, min_doc_freq_threshold)
	print("\nTOP {}-GRAMS BY UNIGRAM SURPRISE".format(n))
	print_top_n(by_unigram_surprise, num_to_print)
	print("\nTOP {}-GRAMS BY BIGRAM SURPRISE".format(n))
	print_top_n(by_bigram_surprise, num_to_print)

def collocates_analysis(ng_model):
	"""
	A term's collocates are terms appearing in a given window around the term,
	across a text corpus.

	The user defines the window shape as a mask of `1`s, `0`s and `x`s where:
	- "x" is the term
	- "1" denotes positions included in the collocates count
	- "0" denotes positions NOT included in the count

	For example, the mask `110x011` will count words that are within 3 words of the term,
	but are not immediately adjacent to it. The mask `xx1` will count words that occur
	immediately after a given 2-gram, equivalent to a trigram model.
	"""
	mask = user_create_mask()
	collocates = ng_model.build_collocates_from_mask(mask=mask, min_count_threshold=100)
	print('normalizing...')
	normalized_collocates = {k: normalize(v) for k, v in collocates.items()}
	print('comparing to baseline...')
	relative_collocates = {k: keywise_quotients(v, ng_model.unigram_model) for k, v in normalized_collocates.items()}
	explore_nested_dict(relative_collocates, 25)

def laugh_rate_analysis(df, n, count_threshold, num_to_print):
	"""
	For a given ngram size, find which ngrams were followed by <Laughter>
	at the highest rates.
	"""

	laugh_ngram_counts = sum_counters(df['laugh_lines'].apply(lambda x: ngram_counts_for_lines(x, n)))
	overall_ngram_counts = sum_counters(df['lines'].apply(lambda x: ngram_counts_for_lines(x, n)))
	laugh_rates = keywise_rates_of_condition( condition_counts=laugh_ngram_counts,
												overall_counts=overall_ngram_counts,
												count_threshold=count_threshold
											)

	print('\nTOP LAUGH RATES (min {} occurrences):'.format(count_threshold))
	for word, quotient in top_n(laugh_rates, num_to_print):
		print(word + '\t' + str(quotient))

	# print('\nLOWEST NONZERO LAUGH RATES (min {} occurrences):'.format(count_threshold))
	# for word, quotient in bottom_n(laugh_rates, num_to_print):
	# 	print(word + '\t' + str(quotient))


if __name__ == '__main__':
	from pprint import pprint
	from utilities import librarian
	df = librarian.load_dataframe(truncate=2500)	# Optional: clip dataframe for testing
	ng_model = NgramModel(df)

	### PROCEDURES (uncomment to run)
	
	#surprise_analysis(ng_model, n=3, min_count_threshold=3, min_doc_freq_threshold=5)
	#collocates_analysis(ng_model)
	laugh_rate_analysis(df, n=3, count_threshold=20, num_to_print=20)





	









	

