import sys
sys.path.append('.')
import random
from ted import *
from utilities import librarian

# LOAD DATA
df = librarian.load_dataframe()

mask = [True, False, True]
all_lines = flatten_list(df['lines'])
skipgram_tree = build_skipgram_tree(mask, all_lines)
skipgram_counts = {k: sum(list(v.values())) for k, v in skipgram_tree.items()}
common_skipgrams = above_threshold(skipgram_counts, 1)

# measure entropy among the n most likely options
skipgram_entropies = {k: entropy(dict(bottom_n(skipgram_tree[k], 20)))
						for k in skipgram_tree if k in common_skipgrams}

tags = get_tags('tag_counts.txt')

while True:
	chosen_tag = user_choose_from_list(tags[:20])
	line_lists = [lines for i, lines in enumerate(df['lines']) if chosen_tag in df['tags'][i]]
	lines = flatten_list(line_lists)
	line = random.choice(lines)
	show_slot_entropies(line, skipgram_entropies, skipgram_tree)