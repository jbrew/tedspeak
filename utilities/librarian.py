import pandas as pd
import re

def load_dataframe(cutoff=2500):
	df1 = pd.read_csv('data/ted_main.csv')
	df2 = pd.read_csv('data/transcripts.csv')
	df = pd.merge(left=df1, right=df2, how='left', left_on='url', right_on='url')
	df = df.head(cutoff)	# Optional: clip dataframe for testing
	df['tags'] = df['tags'].apply(lambda x: set(eval(x)))
	df['lines'] = df['transcript'].apply(lambda x: lines_from_text(x))
	return df

exclude_set = set(['♫','♪'])	# exclude lines with any of these tokens

def lines_from_text(text):
	if not isinstance(text, str): 	# catch nan values
		return []
	lines = clean(text).split('. ')
	return [line.strip() for line in lines if len(line) > 0 and not any_in_line(line, exclude_set)]

def any_in_line(line, exclude_set):
	for ch in line:
		if ch in exclude_set:
			return True
	return False

def clean(text: str):
	text = text.replace(',', '')
	text = text.replace('"', '')
	text = text.replace('?', '.')
	text = text.replace('!', '.')
	text = text.replace(':', '.')
	text = text.replace('.','. ')
	text = handle_parentheticals(text)
	return text

def handle_parentheticals(text: str):
	text = text.replace("(Applause)", ". <Applause>. ")
	text = text.replace("(Laughter)", ". <Laughter>. ")
	return re.sub(r'\(.*?\)', '', text)		# remove all other parentheticals

def get_tags(fpath):
	"""fpath: path to a tab delimited file of common tags and their counts"""
	with open(fpath) as f:
		tag_counts = [line.strip().split('\t') for line in f.readlines()]
	return [tag for tag, __ in tag_counts]