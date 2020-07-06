from utilities.librarian import flatten_list

from utilities.dictionary import (
	enter_nested_item,
	keywise_quotients,
	normalize
)

from utilities.ngram_utils import (
	ngrams_for_line,
	split_ngrams_for_sequence,
	ngram_counts_for_lines,
	model_from_counts,
	tf_idf
)

### CONSTANTS ###

OOV_PENALTY = 0.0000000000000001

class NgramModel(object):

	def __init__(self, df, max_ng_size):
		self.lines_by_doc = df['lines']
		self.lines = flatten_list(df['lines'])
		self.max_ng_size = max_ng_size

		self.ng_counts = []		# ngram counts by size
		self.ng_rates = []		# ngram rates by size
		self.ng_models = []		# P(last token | all preceding tokens)
		self.populate_ngrams()

		self.tf_dicts = []		# term frequency
		self.df_dicts = []		# doc frequency
		self.tfidf_dicts = []	# term frequency / inverse document frequency
		self.populate_tfidf()

	def get_ngram_counts(self, n):
		return self.ng_counts[n-1]

	def get_ngram_rates(self, n):
		return self.ng_rates[n-1]

	def get_ngram_model(self, n):
		return self.ng_models[n-1]

	def populate_ngrams(self):
		for n in range(1, self.max_ng_size+1):
			print('building {}-gram models...'.format(n))
			counts = ngram_counts_for_lines(self.lines, n)
			rates = normalize(counts)
			model = model_from_counts(counts, n)
			self.ng_counts.append(counts)
			self.ng_rates.append(rates)
			self.ng_models.append(model)

	def populate_tfidf(self):
		for n in range(1, self.max_ng_size+1):
			print('tfidf {}-grams...'.format(n))
			tf, df, tfidf = tf_idf(self.lines_by_doc, n, 1)
			self.tf_dicts.append(tf)
			self.df_dicts.append(df)
			self.tfidf_dicts.append(tfidf)

	def ngram_likelihood(self, line, n):
		likelihood = 1
		tokens = line.split()
		for ngram in split_ngrams_for_sequence(tokens, n):
			pass
			# TODO: complete

	def unigram_likelihood(self, line):
		likelihood = 1
		for token in line.split():
			unigram_model = self.get_ngram_rates(1)
			if token in unigram_model:
				likelihood *= unigram_model[token]
			else:
				likelihood *= OOV_PENALTY
		return likelihood

	def bigram_likelihood(self, line):
		model = self.get_ngram_model(2)
		tokens = line.split()
		likelihood = self.unigram_likelihood(tokens[0])
		for w1, w2 in split_ngrams_for_sequence(tokens, 2):
			next_tok_likelihood = model[w1][w2] if w1 in model and w2 in model[w1] else OOV_PENALTY
			likelihood *= next_tok_likelihood
		return likelihood

	def build_collocates_from_mask(self, mask, min_doc_freq_threshold=10):
		d = {}
		target_start = mask.index('X')
		target_end = len(mask) - mask[::-1].index('X')
		for line in self.lines:
			ngrams = ngrams_for_line(line, len(mask))
			for ngram in ngrams:
				tokens = ngram.split()
				key = " ".join(tokens[target_start:target_end])
				for mask_value, token in zip(mask, tokens):
					unigram_counts = self.get_ngram_counts(1)
					if mask_value is True and unigram_counts[token] >= min_doc_freq_threshold:
						enter_nested_item(d, key, token, 1)
		return d

	def ngrams_by_unigram_and_bigram_surprise(self, n, min_count_threshold, min_doc_freq_threshold):
		counts = self.ng_counts[n-1]
		rates = self.ng_rates[n-1]
		doc_freqs = self.df_dicts[n-1]

		common_rates = {ng: rate for ng, rate in rates.items()
						if doc_freqs[ng] > min_doc_freq_threshold 
						and counts[ng] >= min_count_threshold}

		by_unigram_likelihood = {ng: self.unigram_likelihood(ng) for ng in rates}
		by_bigram_likelihood = {ng: self.bigram_likelihood(ng) for ng in counts}
		by_unigram_surprise = keywise_quotients(common_rates, by_unigram_likelihood)
		by_bigram_surprise = keywise_quotients(common_rates, by_bigram_likelihood)

		return by_unigram_surprise, by_bigram_surprise






