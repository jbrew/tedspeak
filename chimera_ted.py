import random
import itertools

import spacy

from utilities import librarian

"""
Generalization of the "chimera" method described here:
https://www.foundpoetryreview.com/blog/oulipost-16-chimera/
"""

def replace_mask_token_in_order(doc, mask_token, source_tokens):
	"""Replace all occurrences of the given masked token with 
	tokens from the iterable source_tokens"""
	result = []
	deck = itertools.cycle(source_tokens)
	for token in doc:
		if token == mask_token:
			replacement = deck.__next__().upper()
			result.append(replacement)
		else:
			result.append(token)
	return result

# VERB CASE

def mask_verbs(doc):
	return [apply_mask_if_verb(token) for token in doc]

def apply_mask_if_verb(token):
	return '<VERB>' if token.pos_ == 'VERB' else token.text

def verbs_for_doc(doc):
	return [token.text.upper() for token in doc if token.pos_ == "VERB"]

def replace_verbs(target_doc, source_doc, pos):
	masked_target = mask_verbs(target_doc)
	source_verbs = verbs_for_doc(source_doc)
	return replace_mask_token_in_order(masked_target, 'VERB', source_verbs)

# POS CASE

def mask_by_pos(doc, pos):
	return [apply_mask_if_pos_match(token, pos) for token in doc]

def apply_mask_if_pos_match(token, pos_to_match):
	return '<{}>'.format(pos_to_match) if token.pos_ == pos_to_match else token.text

def tokens_matching_pos_for_doc(doc, pos):
	return [token.text for token in doc if token.pos_ == pos]

def replace_by_pos(target_doc, source_doc, pos):
	masked_target = mask_by_pos(target_doc, pos)
	source_tokens = tokens_matching_pos_for_doc(source_doc, pos)
	pos_token = '<'+pos+'>'
	return replace_mask_token_in_order(masked_target, pos_token, source_tokens)

# DEPENDENCIES

def deps_for_doc(doc):
	return list(set([token.dep_ for token in doc]))

# returns a map from unique values of dep_ to indices of tokens with that value
def get_dep_to_index_map(doc):
	d = {k: [] for k in deps_for_doc(doc)}
	for i, token in enumerate(doc):
		d[token.dep_].append(i)
	return d

def get_words_for_dep(doc, dep):
	dep_to_index_map = get_dep_to_index_map(doc)
	return [doc[i] for i in dep_to_index_map.get(dep)]

def replace_deps_in_doc(doc, keys_to_replace):
	result = []
	dep_to_index_map = get_dep_to_index_map(doc)
	for token in doc:
		if token.dep_ in keys_to_replace:
			replacement_index = random.choice(dep_to_index_map[token.dep_])
			result.append(doc[replacement_index].text.upper())
		else:
			result.append(token.text.lower())
	return result

def randomly_replace_all_deps(doc):
	return replace_deps_in_doc(doc, set(deps_for_doc(doc)))

if __name__ == '__main__':
	nlp = spacy.load("en_core_web_sm")	# or en_core_web_md or en_core_web_lg
	print('loaded')
	df = librarian.load_dataframe(truncate=250)
	df['transcript'] = df['transcript'].apply(lambda x: librarian.clean_transcript(x))
	texts = list(df['transcript'])
	#random.shuffle(texts)

	text1, text2 = texts[:2]
	doc1 = nlp(text1)
	doc2 = nlp(text2)

	nouns_replaced = replace_by_pos(doc1, doc2, "NOUN")
	adjectives_replaced = replace_by_pos(doc1, doc2, "ADJ")

	dobj_replaced = replace_deps_in_doc(doc1, ['dobj'])

	all_deps_replaced = randomly_replace_all_deps(doc1)

	replace_set = set(['xcomp', 'dobj'])
	deps_replaced = replace_deps_in_doc(doc1, replace_set)

	print(" ".join(adjectives_replaced))


