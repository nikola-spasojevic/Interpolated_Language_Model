import pickle
from heapq import heappush, heappushpop, heappop
from collections import deque

CAPACITY=5
N_GRAM=3

class JobTitleRecommender:
	def __init__(self, v_tree_dir='bin/v_tree.pkl', trie_dir='bin/trie.pkl'):
		# vocabulary tree for ngram likelihood estimates (next word prediction)
		with open(v_tree_dir, 'rb') as pickle_in:
			self.v_tree = pickle.load(pickle_in, encoding='utf8')
		# trie for autocomplete
		with open(trie_dir, 'rb') as pickle_in:
			self.trie = pickle.load(pickle_in, encoding='utf8')

	# Retrive top CAPACITY results using unigram scores
	def auto_complete(self, prefix, capacity=CAPACITY):
		min_heap = []
		trie_generator = self.trie.all_words_beginning_with_prefix(prefix)
		while True:
			try:
				val = next(trie_generator)
				if len(min_heap) < capacity:
					heappush(min_heap, (self.v_tree.get_likelihood(val), val))
				else:
					heappushpop(min_heap, (self.v_tree.get_likelihood(val), val))
			except StopIteration:
				break
		return sorted([heappop(min_heap) for i in range(len(min_heap))], reverse=True)

	# Retrive top CAPACITY results from the V-ary tree
	# likelihoods are precalculated in the Language Model module
	# Based upon Maximum Likelihood, a sequence of words (context) has a set of 'next word' results.
	# This is used to predict the next possible words based on training data
	def predict_next_word(self, text, capacity=CAPACITY):
		text = '<s> <s> '+text
		text = ' '.join(text.split())
		min_heap = []
		context = tuple(text.split()[-N_GRAM+1:])
		l = self.v_tree.word_prediction(context)
		for word_lklhd in l:
			if len(min_heap) < capacity:
				heappush(min_heap, word_lklhd)
			else:
				heappushpop(min_heap, word_lklhd)
		return sorted([heappop(min_heap) for i in range(len(min_heap))], reverse=True)

def main():
	job_title_recommender = JobTitleRecommender()
	# print(job_title_recommender.auto_complete('ae'))
	print(job_title_recommender.predict_next_word('java'))

if __name__ == "__main__":
   main()