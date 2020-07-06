import math
from collections import Counter
from utilities.dictionary import (
	sum_counters,
	enter_nested_item,
	normalize
)

### GET NGRAMS ###

def ngrams_for_sequence(tokens, n):
	return [' '.join(tokens[start:start+n]) for start in range(len(tokens) + 1 - n)]

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

def model_from_counts(ngram_counts, n):
	model = {}
	for ngram, count in ngram_counts.items():
		tokens = ngram.split()
		context, observation = ' '.join(tokens[:-1]), tokens[-1]
		enter_nested_item(model, context, observation, count)
	return {k: normalize(v) for k, v in model.items()}

def build_ngram_model(lines, n):
	ngram_counts = ngram_counts_for_lines(lines, n)
	return model_from_counts(ngram_counts, n)

### TFIDF ###

def tf_idf(lines_by_doc, ngram_size, min_doc_freq_threshold=1):
	total_docs = len(lines_by_doc)
	term_frequencies = list(map(lambda x: ngram_counts_for_lines(x, ngram_size), lines_by_doc))
	doc_frequencies = sum_counters([{k: 1 for k in counter} for counter in term_frequencies])
	tfidfs = list(map(lambda x:
		{k: tf * -math.log(doc_frequencies[k]/total_docs)
		for k, tf in x.items()
		if doc_frequencies[k] >= min_doc_freq_threshold}, term_frequencies)
		)
	return term_frequencies, doc_frequencies, tfidfs