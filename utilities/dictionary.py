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

def sum_counters(counters):
	"""counters: a list of Counter objects"""
	agg = {}
	for counter in counters:
		for k, v in counter.items():
			enter_item(agg, k, v)
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
