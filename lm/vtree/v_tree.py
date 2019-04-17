from collections import defaultdict
from math import log

class VNode:
	def __init__(self, lklhd=log(0.0002016942315449778)):
		self.lklhd = lklhd
		self.children = defaultdict(VNode)

class VTree:
	def __init__(self, unk_score, threshold=0.0002):
		self.root = VNode()
		self.unk_score = unk_score
		self.threshold = threshold

	def insert(self, target_word, lklhd, context=()):
		curr = self.root
		for word in context:
			node = curr.children[word]
			curr = node
		curr.children[target_word].lklhd = lklhd

	def get_likelihood(self, target_word, context=()):
		curr = self.root
		for word in context:
			node = curr.children.get(word)
			if not node:
				# Backoff Model - lower ngram by 1 degree
				c_lst = list(context)
				return self.get_likelihood(target_word, tuple(c_lst[1:]))
		node = curr.children.get(target_word)
		return self.unk_score if not node else node.lklhd

	def print_levels(self):
		for k1, v1 in self.root.children.items():
			print('first_level: {}, {}'.format(k1, v1.lklhd))
			for k2, v2 in v1.children.items():
				print('second_level: {}, {}'.format(k2, v2.lklhd))
				for k3, v3 in v2.children.items():
					print('third_level: {}, {}'.format(k3, v3.lklhd))
			print()

	def word_prediction(self, context=()):
		curr = self.root
		for word in context:
			node = curr.children.get(word)
			if not node:
				# Backoff Model - lower ngram by 1 degree
				c_lst = list(context)
				return self.word_prediction(tuple(c_lst[1:]))
			curr = node
		return [(node.lklhd, word) for word, node in curr.children.items()]
