import sys
sys.path.append('.')
from utilities.dictionary import top_n, bottom_n

translation_table = {'0': False, '1': True, 'X': 'X', 'x': 'X'}

def user_choose_from_list(options):
	for i, option in enumerate(options):
			print(i+1, option)
	selected = input('Choose a number from above\n')
	try:
		return options[int(selected)-1]
	except:
		print('Try again')
		return user_choose_from_list(options)

def user_create_mask():
	user_input = input('Enter mask as 0s, 1s and Xs, e.g. 110x011:\n')
	return [translation_table[character] for character in user_input]

def print_top_n(d, n=20):
	print('\n'.join([str(pair) for pair in top_n(d, n)]))

def print_bottom_n(d, n=20):
	print('\n'.join([str(pair) for pair in bottom_n(d, n)]))

def explore_model(ngram_model):
	while True:
		key = input('Enter word:\n')
		if key in ngram_model:
			print(ngram_model[key])

def explore_nested_dict(d, top_n=20):
	while True:
		key = input('Enter word:\n')
		if key in d:
			print_top_n(d[key], top_n)

