class VNode:
	def __init__(self, lklhd=0.0):
		self.lklhd = lklhd
		self.children = {}

class VTree:
	def __init__(self, unk_score, threshold=0.0002):
		self.root = VNode()
		self.unk_score = unk_score
		self.threshold = threshold

	def insert(self, target_word, lklhd, context=()):
		curr = self.root
		for word in context:
			node = curr.children.get(word)
			if not node:
				node = VNode()
				curr.children[word] = node
			curr = node
		curr.children[target_word] = VNode(lklhd)

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