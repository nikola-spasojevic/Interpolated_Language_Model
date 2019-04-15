class VNode:
	def __init__(self, lklhd=0.0):
		self.lklhd = lklhd
		self.children = {}

	def all_words(self, word=''):
		if not self.children:
			yield word

		for word, child in self.children.items():
			yield from child.all_words(word)

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
			curr = node
		node = curr.children.get(target_word)
		return self.unk_score if not node else node.lklhd

	def word_prediction(self, context=()):
		curr = self.root
		print(context)
		for word in context:
			node = curr.children.get(word)
			if not node:
				print('Backoff')
				# Backoff Model - lower ngram by 1 degree
				c_lst = list(context)
				return self.word_prediction(tuple(c_lst[1:]))

		return node.children.items()
