from textpreprocessing.text_preprocessing import TextPreprocessing
from lm.language_model import NgramLanguageModel
from lm.trie.trie import Trie, TrieNode
from jobtitlerecommender.job_title_recommender import JobTitleRecommender
import sys
import os

def main():
	# NgramLanguageModel.likelihoods_gen(TRAIN_CORPUS_DIR='bin/train_corpus.pkl', TEST_CORPUS_DIR='bin/test_corpus.pkl')
	job_title_recommender = JobTitleRecommender()
	print("Input Job Title: ")
	while True:		
		var = input('\n')
		result_list = []
		if var:
			if var[-1] == ' ':
				result_list.extend(job_title_recommender.predict_next_word(var[:-1]))
				print('\n{:<20}{:>10}'.format('Next Word Prediction', 'Score'))
			else:
				var = var.split(' ')
				result_list.extend(job_title_recommender.auto_complete(var[-1]))
				print('\n{:<20}{:>10}'.format('Autocomplete', 'Score'))
			for score, word in result_list:
				print('{:<20} |{:>1.10f}'.format(word, score))

if __name__ == "__main__":
   main()