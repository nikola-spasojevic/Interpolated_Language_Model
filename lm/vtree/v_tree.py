class VNode:
	def __init__(self, lklhd=0.0):
		self.lklhd = lklhd
		self.children = {}

class VTree:
	def __init__(self, threshold=0.0002):
		self.root = VNode()
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

	def get_likelihood(self, word, context=None):
		curr = self.root
		if not context:
			unigram = curr.children.get(word)
			return unigram.lklhd if unigram else None
		for w in context:
			node = curr.children.get(word)
			if not node:
				# lower model by 1 degree - backoff model
				return self.get_likelihood(word, context[1:])


