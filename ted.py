import re
import math
import pandas as pd
from collections import Counter

from models.ngrams import (
	NgramModel,
	ngram_counts_for_lines,
	ngrams_for_line,
	flatten_list,
	tf_idf
)

from models.skipgrams import (
	skipgram_counts_for_lines,
	apply_mask_to_ngram,
	mask_for_skipgram,
	entries_matching_skipgram
)
from utilities.dictionary import (
	sum_counters,
	keywise_quotients,
	normalize,
	keywise_rates_of_condition
)
from utilities.command_line import *
from utilities import librarian
from resources.stopwords import stopwords
from resources.tag_counts import tag_counts

##############################################

### PROCEDURES ###

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

def laugh_rate_analysis(df, n, count_threshold, num_to_print):

	laugh_ngram_counts = sum_counters(df['laugh_lines'].apply(lambda x: ngram_counts_for_lines(x, n)))
	overall_ngram_counts = sum_counters(df['lines'].apply(lambda x: ngram_counts_for_lines(x, n)))
	laugh_rates = keywise_rates_of_condition( condition_counts=laugh_ngram_counts,
												overall_counts=overall_ngram_counts,
												count_threshold=count_threshold
											)

	print('\nLOWEST NONZERO LAUGH RATES (min {} occurrences):'.format(count_threshold))
	for word, quotient in bottom_n(laugh_rates, num_to_print):
		print(word + '\t' + str(quotient))

	print('\nTOP LAUGH RATES (min {} occurrences):'.format(count_threshold))
	for word, quotient in top_n(laugh_rates, num_to_print):
		print(word + '\t' + str(quotient))

def tag_tfidfs(df, n, min_doc_freq_threshold):
	from pprint import pprint

	tags = [tag for tag, count in top_n(tag_counts, 50)[10:]]
	total_docs = len(tags)	
	ngram_counts_by_tag = {}

	for tag in tags:
		print(tag)
		line_lists = [lines for i, lines in enumerate(df['lines']) if tag in df['tags'][i]]
		ngram_counts_for_tag = sum_counters([ngram_counts_for_lines(line_list, n) for line_list in line_lists])
		ngram_counts_by_tag[tag] = ngram_counts_for_tag
	
	doc_frequencies = sum_counters([{k: 1 for k in counter} for counter in ngram_counts_by_tag.values()])

	tfidfs = {tag: {k: tf * -math.log(doc_frequencies[k]/total_docs)
				for k, tf in term_frequencies.items()
				if doc_frequencies[k] >= min_doc_freq_threshold} for tag, term_frequencies in ngram_counts_by_tag.items()}

	for k, v in tfidfs.items():
		print(k)
		pprint(top_n(v, 10))
		print()


BLANK_TOKEN = '___'

def mask_content_words(tokens):
	return [mask_if_content_word(token) for token in tokens]

def mask_if_content_word(token):
	return token if token in stopwords else BLANK_TOKEN

def mask_stopwords(tokens):
	return [mask_if_stopword(token) for token in tokens]

def mask_if_stopword(token):
	return token if token not in stopwords else BLANK_TOKEN

def fill_content_words(masked, ng_model):
	tokens = masked.split()
	# TODO

def mask_tokens_if_in_set(tokens, set_to_mask):
	return [mask_if_in_set(token, set_to_mask) for token in tokens]

def mask_if_in_set(token, set_to_mask):
	return BLANK_TOKEN if token in set_to_mask else token

def mask_all_lines_if_in_set(lines, set_to_mask):
	return [mask_tokens_if_in_set(ngrams_for_line(line, 1), set_to_mask) for line in lines]


def stopword_analysis(df):
	lines = flatten_list(df['lines'])
	
	for line in lines[:100]:
		ngrams = ngrams_for_line(line, 5)
		for ngram in ngrams[:5]:
			print(ngram)
			print(" ".join(mask_content_words(ngram.split())))
			print()

def top_n_keys(d, n):
	return [k for k, v in top_n(d, n)]

def mask_tfidfs(df):
	lines_by_title = dict(zip(df['title'], df['lines']))
	term_frequencies, doc_frequencies, tfidfs = tf_idf(df['lines'], ngram_size=1, min_doc_freq_threshold=2)
	tfidfs_by_title = dict(zip(df['title'], tfidfs))

	n = 100
	import random
	title = random.choice(df['title'])
	tfidfs = tfidfs_by_title[title]
	top_keywords = set(top_n_keys(tfidfs, 100)) - stopwords
	for line in lines_by_title[title]:
		tfidf_masked_line = ' '.join(mask_tokens_if_in_set(line.lower().split(), top_keywords))
		stopword_masked_line = ' '.join(mask_content_words(line.lower().split()))
		print(line)
		print(tfidf_masked_line)
		print(stopword_masked_line)
		print()


if __name__ == '__main__':
	from pprint import pprint
	df = librarian.load_dataframe(truncate=250)	# Optional: clip dataframe for testing
	ng_model = NgramModel(df)


	# print(len(ng_model.unigram_model))
	# print(len(ng_model.bigram_model_alt))
	# print(len(ng_model.trigram_model))

	# for k, v in list(ng_model.trigram_model.items())[:100]:
	# 	print(k)

	
	### PROCEDURES (uncomment to run)
	#surprise_analysis(ng_model, n=6, min_count_threshold=3, min_doc_freq_threshold=5)
	#collocates_analysis(ng_model)
	#laugh_rate_analysis(df, 3, 20, 100)
	#tag_tfidfs(df, n=1, min_doc_freq_threshold=15)

	## TALK TFIDFS







	









	

