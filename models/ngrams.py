import math
from collections import Counter
from typing import List, Sequence

from utilities.dictionary import (
	sum_counters,
	enter_nested_item,
	keywise_quotients,
	normalize
)

### CONSTANTS ###

OOV_PENALTY = 0.0000000000000001

### GET NGRAMS ###

def ngrams_for_sequence(tokens, n):
	return [' '.join(tokens[start:start+n]) for start in range(len(tokens) - n)]

def split_ngrams_for_sequence(tokens, n):
	return [tokens[start:start+n] for start in range(len(tokens) - n)]

def ngrams_for_line(line, n):
	words = line.lower().split()
	return ngrams_for_sequence(words, n)

def ngram_counts_for_line(line, n):
	return Counter(ngrams_for_line(line, n))

def ngram_counts_for_lines(lines, n):
	return sum_counters([ngram_counts_for_line(line, n) for line in lines])

### BUILD MODELS ###

def build_unigram_model(lines):
	return normalize(ngram_counts_for_lines(lines, 1))

def build_bigram_model(lines):
	bigram_counts = ngram_counts_for_lines(lines, 2)
	return bigram_model_from_counts(bigram_counts)

def bigram_model_from_counts(bigram_counts):
	model = {}
	for bigram, count in bigram_counts.items():
		w1, w2 = bigram.split()
		enter_nested_item(model, w1, w2, count)
	return {k: normalize(v) for k, v in model.items()}

### LINE OPERATIONS ###

def get_laugh_lines(lines: List[str]):
	return [start for start, end in zip(lines[:-1], lines[1:]) if end == '<Laughter>']

def get_applause_lines(lines: List[str]):
	return [start for start, end in zip(lines[:-1], lines[1:]) if end == '<Applause>']

def lines_for_tag(df, tag):
	return df.loc[tag in df['tags']]

def flatten_list(list_of_lists):
	return [item for sublist in list_of_lists for item in sublist]

### TFIDF ###

def tf_idf(lines, ngram_size, min_doc_freq_threshold=1):
	total_docs = len(lines)
	term_frequencies = list(map(lambda x: ngram_counts_for_lines(x, ngram_size), lines))
	doc_frequencies = sum_counters([{k: 1 for k in counter} for counter in term_frequencies])
	tfidfs = list(map(lambda x:
		{k: tf * -math.log(doc_frequencies[k]/total_docs)
		for k, tf in x.items()
		if doc_frequencies[k] >= min_doc_freq_threshold}, term_frequencies)
		)
	return term_frequencies, doc_frequencies, tfidfs


class NgramModel(object):

	def __init__(self, lines_by_doc):
		self.lines_by_doc = lines_by_doc
		self.lines = flatten_list(lines_by_doc)

		self.unigram_counts = ngram_counts_for_lines(self.lines, 1)
		self.unigram_model = normalize(self.unigram_counts)
		
		self.bigram_counts = ngram_counts_for_lines(self.lines, 2)
		self.bigram_model = bigram_model_from_counts(self.bigram_counts)
		
		self.collocates = {} # TODO: key on masks

	def unigram_likelihood(self, line):
		likelihood = 1
		for token in line.split():
			if token in self.unigram_model:
				likelihood *= self.unigram_model[token]
			else:
				likelihood *= OOV_PENALTY
		return likelihood

	def bigram_likelihood(self, line):
		model = self.bigram_model
		tokens = line.split()
		likelihood = self.unigram_likelihood(tokens[0])
		for w1, w2 in split_ngrams_for_sequence(tokens, 2):
			next_tok_likelihood = model[w1][w2] if w1 in model and w2 in model[w1] else OOV_PENALTY
			likelihood *= next_tok_likelihood
		return likelihood

	def ngrams_by_unigram_and_bigram_surprise(self, n, min_count_threshold, min_doc_freq_threshold):
		counts = ngram_counts_for_lines(self.lines, n)
		rates = normalize(counts)
		__, doc_frequencies, __ = tf_idf(self.lines_by_doc, n)

		common_rates = {ng: rate for ng, rate in rates.items()
					if doc_frequencies[ng] > min_doc_freq_threshold 
					and counts[ng] >= min_count_threshold}

		by_unigram_likelihood = {ng: self.unigram_likelihood(ng) for ng in rates}
		by_bigram_likelihood = {ng: self.bigram_likelihood(ng) for ng in counts}
		by_unigram_surprise = keywise_quotients(common_rates, by_unigram_likelihood)
		by_bigram_surprise = keywise_quotients(common_rates, by_bigram_likelihood)

		return by_unigram_surprise, by_bigram_surprise


	def build_collocates_from_mask(self, mask, min_count_threshold=100):
		d = {}
		x_index = mask.index('X')
		for line in self.lines:
			ngrams = ngrams_for_line(line, len(mask))
			for ngram in ngrams:
				tokens = ngram.split()
				x_token = tokens[x_index]
				for mask_value, token in zip(mask, tokens):
					if mask_value is True and self.unigram_counts[token] > min_count_threshold:
						enter_nested_item(d, x_token, token, 1)
		return d


def quotients_for_condition(condition, baseline, count_threshold=1):
	"""
	returns a dictionary:
		{ngram: P(ngram|condition)/P(ngram)}
	"""
	summed_counts_for_base = sum_counters(baseline)
	normalized_base = normalize(summed_counts_for_base)
	summed_counts_for_cond = sum_counters(condition)
	normalized_cond = normalize(summed_counts_for_cond)

	above_threshold_base = {k: v for k, v in normalized_base.items()
							if summed_counts_for_base[k] >= count_threshold}
	
	above_threshold_cond = {k: v for k, v in normalized_cond.items()
							if summed_counts_for_cond[k] >= count_threshold}
	
	return keywise_quotients(above_threshold_cond, above_threshold_base)

def condition_rates_by_ngram(condition, overall, count_threshold):
	"""e.g. P(laugh|ngram)"""
	summed_counts_for_cond = sum_counters(condition)
	summed_counts_overall = sum_counters(overall)
	above_threshold_cond = {k: v for k, v in summed_counts_for_cond.items() if summed_counts_overall[k]>count_threshold}
	return {k: v/summed_counts_overall[k] for k, v in above_threshold_cond.items()}




