import sys
sys.path.append('.')
import random

from models.skipgrams import (
	build_skipgram_tree,
	show_slot_entropies
)

from utilities.dictionary import (
	entropy,
	top_n,
	bottom_n,
	above_threshold
)
from models.ngrams import (
	flatten_list
)
from utilities import librarian
from utilities import command_line
from resources.tag_counts import tag_counts


if __name__ == "__main__":
	df = librarian.load_dataframe(truncate=250)

	mask = [True, False, True]
	all_lines = flatten_list(df['lines'])
	skipgram_tree = build_skipgram_tree(mask, all_lines)
	skipgram_counts = {k: sum(list(v.values())) for k, v in skipgram_tree.items()}
	common_skipgrams = above_threshold(skipgram_counts, 1)

	# measure entropy among the n most likely options
	skipgram_entropies = {k: entropy(dict(bottom_n(skipgram_tree[k], 20)))
							for k in skipgram_tree if k in common_skipgrams}

	tags = [tag for tag, count in top_n(tag_counts, 50)[10:]]	# ranks 11 to 50

	# User chooses tag, computer chooses random line from talk with that tag
	while True:
		chosen_tag = command_line.user_choose_from_list(tags[:20])
		line_lists = [lines for i, lines in enumerate(df['lines']) if chosen_tag in df['tags'][i]]
		lines = flatten_list(line_lists)
		line = random.choice(lines)
		show_slot_entropies(line, skipgram_entropies, skipgram_tree)