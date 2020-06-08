import re
import pandas as pd
from collections import Counter

### TEXT FETCHING AND CLEANING ###

def lines_from_transcript(text):
	if not isinstance(text, str): 	# catch nan values
		return []
	lines = clean(text).split('. ')
	return [line.strip() for line in lines if len(line) > 0]	

def clean(text):
	text = text.replace('.','. ')
	text = text.replace(',', '')
	text = text.replace('"', '')
	text = text.replace('?', '.')
	text = text.replace('!', '.')
	text = handle_parens(text)
	return text

def handle_parens(text):
	text = text.replace("(Applause)", ". <Applause>. ")
	text = text.replace("(Laughter)", ". <Laughter>. ")
	return re.sub(r'\(.*?\)', '', text)		# ignore all parens but applause and laughter

### NGRAMS ###

def ngram_counts_for_lines(lines, n):
	return sum([ngram_counts_for_line(line, n) for line in lines], Counter())

def ngram_counts_for_line(line, n):
	words = line.lower().split()
	return Counter([' '.join(words[start:start+n]) for start in range(len(words) - n)])

### LINES ###

def laugh_lines(lines):
	return [start for start, end in zip(lines[:-1], lines[1:]) if end == '<Laughter>']

def applause_lines(lines):
	return [start for start, end in zip(lines[:-1], lines[1:]) if end == '<Applause>']

def sum_counters(column):
	"""column: a list of Counter objects"""
	return sum(column, Counter())

### DICTIONARY OPERATIONS ###

def normalize(d):
	total = sum(d.values())
	return {k: v/total for k, v in d.items()}

def keywise_quotients(d, baseline):
	return {k: d[k] / baseline[k] for k in d if k in baseline}

def top_n(d, n):
	return sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]

def above_threshold(d, threshold):
	return {k: v for k, v in d.items() if v >= threshold}

### MACRO ###

def quotients_for_field(df, field_name, baseline_name, count_threshold):
	summed_counts_for_baseline = sum_counters(df[baseline_name])
	summed_counts_for_field = sum_counters(df[field_name])

	baseline_clearing_threshold = above_threshold(summed_counts_for_baseline, count_threshold)
	field_clearing_threshold = above_threshold(summed_counts_for_field, count_threshold)

	normalized_baseline = normalize(baseline_clearing_threshold)
	normalized_field = normalize(field_clearing_threshold)

	return keywise_quotients(normalized_field, normalized_baseline)

if __name__ == '__main__':

	df1 = pd.read_csv('data/ted_main.csv')
	df2 = pd.read_csv('data/transcripts.csv')
	df3 = pd.merge(left=df1, right=df2, how='left', left_on='url', right_on='url')

	df3['lines'] = df3['transcript'].apply(lambda x: lines_from_transcript(x))
	df3['laugh_lines'] = df3['lines'].apply(lambda x: laugh_lines(x))
	df3['applause_lines'] = df3['lines'].apply(lambda x: applause_lines(x))

	df3 = df3.head(1000)		# uncomment for faster testing

	count_threshold = 10		# words must occur this many times
	num_to_display = 20

	### WORD ANALYSIS ###

	"""
	df3['all_1grams'] = df3['lines'].apply(lambda x: ngram_counts_for_lines(x, 1))	
	df3['laugh_1grams'] = df3['laugh_lines'].apply(lambda x: ngram_counts_for_lines(x, 1))
	df3['applause_1grams'] = df3['applause_lines'].apply(lambda x: ngram_counts_for_lines(x, 1))

	laugh_quotients = quotients_for_field(df3, 'laugh_1grams', 'all_1grams', count_threshold)
	laugh_words = top_n(laugh_quotients, num_to_display)

	applause_quotients = quotients_for_field(df3, 'applause_1grams', 'all_1grams', count_threshold)
	applause_words = top_n(applause_quotients, num_to_display)

	print('\nLAUGH WORDS:')
	for word, quotient in laugh_words:
		print(word + '\t' + str(quotient))

	print('\nAPPLAUSE WORDS:')
	for word, quotient in applause_words:
		print(word + '\t' + str(quotient))
	"""

	### 2GRAM ANALYSIS ###
	"""
	count_threshold = 10		# ngrams must occur this many times
	num_to_display = 100

	print('counting ngrams')
	df3['all_2grams'] = df3['lines'].apply(lambda x: ngram_counts_for_lines(x, 2))
	print('counting laugh ngrams')
	df3['laugh_2grams'] = df3['laugh_lines'].apply(lambda x: ngram_counts_for_lines(x, 2))

	laugh_quotients = quotients_for_field(df3, 'laugh_2grams', 'all_2grams', count_threshold)
	laugh_2grams = top_n(laugh_quotients, num_to_display)

	print('\nLAUGH 2GRAMS:')
	for word, quotient in laugh_2grams:
		print(word + '\t' + str(quotient))
	"""
	### NGRAM ANALYSIS ###

	n = 3
	count_threshold = 5
	num_to_display = 100

	print('finding ngrams in all lines...')
	df3['all_ngrams'] = df3['lines'].apply(lambda x: ngram_counts_for_lines(x, n))
	print('finding ngrams in laugh lines...')
	df3['laugh_ngrams'] = df3['laugh_lines'].apply(lambda x: ngram_counts_for_lines(x, n))

	print('finding quotients...')
	laugh_quotients = quotients_for_field(df3, 'laugh_ngrams', 'all_ngrams', count_threshold)
	to_display = top_n(laugh_quotients, num_to_display)

	print('\nTOP NGRAMS:')
	for word, quotient in to_display:
		print(word + '\t' + str(quotient))









