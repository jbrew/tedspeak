import sys
sys.path.append('.')

import spacy
from utilities import librarian
from utilities.dictionary import (
	enter_nested_item,
	sum_counters,
	sum_nested_counters,
	top_n
	)

"""
Experimental script using Spacy to track what kinds of tokens
appear in particular syntactic contexts, e.g. when the syntactic
ancestors of a token are [NOUN, VERB], or when it is a verba and 
has a DOBJ dependency relationship to its left child. Idea is to 
make better-educated guesses about what terms are interchangeable.
"""

def syntactic_subtree(token, doc):
	return doc[token.left_edge.i:token.right_edge.i+1]

def text_for_slice(doc_slice):
	return ''.join([token.text_with_ws for token in doc_slice])

def prepositional_phrases_for_doc(doc):
	return [syntactic_subtree(token, doc) for token in doc if token.dep_ == 'prep']

def roots_for_doc(doc):
	return [token for token in doc if token.dep_ == 'ROOT']

def left_deps(tok):
	return [child.dep_ for child in tok.lefts]

def right_deps(tok):
	return [child.dep_ for child in tok.rights]

def get_ancestors(tok, num_ancestors=1):
	if num_ancestors == 0:
		return []
	if tok.dep_ == 'ROOT':
		return []
	return get_ancestors(tok.head, num_ancestors-1) + [tok.head]

def tokens_by_dep_ancestors(doc, max_context_size=3):
	"""doc: a spacy doc"""
	d = {}
	for tok in doc:
		ancestors = get_ancestors(tok, max_context_size)
		dep_history = [tok.dep_] + [anc.dep_ for anc in ancestors]
		dep_key = " ".join(dep_history)
		enter_nested_item(d, dep_key, tok.text, 1)
	return d

def tokens_by_dep_context(doc):
	"""context includes immediate children and parents"""
	d = {}
	for tok in doc:
		child_tags = [child.dep_.lower() for child in tok.children]
		parent_tag = 'ROOT' if tok.dep_ == 'ROOT' else tok.head.dep_
		dep_key = "{}__{}".format(parent_tag, " ".join(child_tags))
		# family = get_family(tok)
		# family_deps = [tok.dep_] + [member.dep_ for member in family]
		# dep_key = " ".join(family_deps)
		enter_nested_item(d, dep_key, tok.text, 1)
	return d


### TESTS ###

def parse_trees_for_first_sentence_in_doc(doc):
	for token in doc[:20]:
		subtree = syntactic_subtree(token, doc)
		if len(subtree) > 1:
			print('TOKEN: {}'.format(token))
			print(text_for_slice(subtree))

def print_prepositional_phrases_in_doc(doc):
	pp1 = prepositional_phrases_for_doc(doc)
	for subtree in pp1:
		if len(subtree) > 1:
			print(text_for_slice(subtree))

if __name__ == '__main__':
	import random
	nlp = spacy.load("en_core_web_sm")
	print('loaded')
	df = librarian.load_dataframe(truncate=25)
	df['transcript'] = df['transcript'].apply(lambda x: librarian.clean_transcript(x))
	texts = list(df['transcript'])

	random.shuffle(texts)

	n = 200
	print('about to compute')
	first_n_docs = [nlp(d) for d in texts[:n]]

	#tokens_by_key = [tokens_by_dep_ancestors(doc) for doc in first_n_docs]
	tokens_by_key = [tokens_by_dep_context(doc) for doc in first_n_docs]

	tree = sum_nested_counters(tokens_by_key)

	# text1, text2 = texts[:2]
	# doc1 = nlp(text1)
	# doc2 = nlp(text2)

	# tree = tokens_by_dep_ancestors(doc1 + doc2)

	largest_entries = sorted(tree.items(), key=lambda x: len(x[1]), reverse=True)
	for k, v in largest_entries[:10]:
		print(k.upper())
		print(top_n(v, 150))



	# roots = roots_for_doc(doc1)
	
	# for r in roots:
	# 	print(r.text.upper())
	# 	dfs(r)
		#print(r.text.upper())
		#print(list(r.subtree))
		#dfs(r)

