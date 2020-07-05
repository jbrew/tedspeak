import math

### DICTIONARY HELPERS ###

def normalize(d):
	total = sum(d.values())
	return {k: v/total for k, v in d.items()}

def top_n(d, n=10):
	return sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]

def bottom_n(d, n=10):
	return sorted(d.items(), key=lambda x: x[1], reverse=False)[:n]

def above_threshold(d, threshold):
	return {k: v for k, v in d.items() if v >= threshold}

def keywise_quotients(d, baseline):
	return {k: d[k] / baseline[k] for k in d if k in baseline}

def keywise_rates_of_condition(condition_counts, overall_counts, count_threshold=1):
	"""e.g. P(laugh|ngram)"""
	common_condition_counts = {k: v for k, v in condition_counts.items() if overall_counts[k]>=count_threshold}
	return {k: v/overall_counts[k] for k, v in common_condition_counts.items()}

def sum_counters(counters):
	"""counters: a list of Counter objects"""
	agg = {}
	for counter in counters:
		for k, v in counter.items():
			enter_item(agg, k, v)
	return agg

def sum_nested_counters(nested_counters):
	"""nested counters: [ {string: {string: number}}, ... ]"""
	agg = {}
	for counter in nested_counters:
		for context, ng_dict in counter.items():
			for ng, count in ng_dict.items():
				enter_nested_item(agg, context, ng, count)
	return agg

def enter_item(d, k, value):
	if k in d:
		d[k] += value
	else:
		d[k] = value

def enter_nested_item(d, k1, k2, value):
	if not k1 in d:
		d[k1] = {k2: value}
	else:
		enter_item(d[k1], k2, value)

def entropy(d):
	"""
	d: a dictionary mapping keys to weights
	"""
	return -1 * sum([d[option] * math.log(d[option]) for option in normalize(d)])
