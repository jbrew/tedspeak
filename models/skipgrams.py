import sys
sys.path.append('.')

from utilities.ngram_utils import (
	ngrams_for_line,
	ngram_counts_for_line,
	ngram_counts_for_lines
)
from utilities.dictionary import(
	top_n,
	bottom_n,
	sum_counters,
	enter_item
)

MASK_TOKEN = '____'

def build_skipgram_tree(mask, lines):
	ngram_size = len(mask)
	ngram_counts = ngram_counts_for_lines(lines, ngram_size)
	skipgram_tree = {}
	for ngram, count in ngram_counts.items():
		skipgram = apply_mask_to_ngram(mask, ngram)
		if not skipgram in skipgram_tree:
			skipgram_tree[skipgram] = {}
		skipgram_tree[skipgram][ngram] = count
	return skipgram_tree

def skipgram_counts_for_lines(lines, mask):
	return sum_counters([skipgram_counts_for_line(line, mask) for line in lines])

def skipgram_counts_for_line(line, mask):
	"""
	Boolean mask of which words to include, e.g.
	[True, True, False, True, True]
	"""
	n = len(mask)
	ngram_counter = ngram_counts_for_line(line, n)
	skipgram_counter = {}
	for k, v in ngram_counter.items():		# collapse across keys
		skip_key = apply_mask_to_ngram(mask, k)
		enter_item(skipgram_counter, skip_key, v)
	return skipgram_counter

def apply_mask_to_ngram(mask, ngram):
	return ' '.join([apply_mask_element(w, visibility) for w, visibility in zip(ngram.split(), mask)])

def apply_mask_element(word, visibility):
	return word if visibility else MASK_TOKEN

def mask_for_skipgram(skipgram):
	return [word != MASK_TOKEN for word in skipgram.split()]

def entries_matching_skipgram(d, skipgram):
	mask = mask_for_skipgram(skipgram)
	return {k: v for k, v in d.items() if apply_mask_to_ngram(mask, k) == skipgram}

def show_slot_entropies(line, skipgram_entropies, skipgram_tree):
	words = line.split()
	ngrams = ngrams_for_line(line, 3)
	for ngram in ngrams:
		skipgram = apply_mask_to_ngram([True, False, True], ngram)
		print(skipgram, skipgram_entropies[skipgram], [x[0].split()[1] for x in top_n(skipgram_tree[skipgram], 10)])

