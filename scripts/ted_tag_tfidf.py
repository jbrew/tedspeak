import sys
sys.path.append('.')
import math

from utilities.dictionary import (
	sum_counters,
	top_n,
)

from models.ngrams import (
	ngram_counts_for_lines,
)
from resources.tag_counts import tag_counts

"""
The idea with this script was to find keywords for different tags in the TED corpus
using tf–idf, but the results were disappointing, which I think is partly to do with
the "documents" (each comprising all talks with a given tag) being too big.
"""

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