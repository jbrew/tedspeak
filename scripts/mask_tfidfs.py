import sys
import math
sys.path.append('.')

from models.ngrams import (
	ngram_counts_for_lines,
	flatten_list,
)

from utilities.dictionary import (
	top_n,
	sum_counters
)
from resources.stopwords import stopwords

"""
Sketch of a utility for a madlibs type game where the recipe is:
1. Identify the keywords in a given talk with tf-idf, a measure of term specificity
2. Mask these keywords
3. Replace them with similar keywords from another talk
"""

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


if __name__ == "__main__":
	from utilities import librarian
	df = librarian.load_dataframe(truncate=250)	# Optional: clip dataframe for testing
	mask_tfidfs(df)