from nltk.lm import KneserNeyInterpolated, WittenBellInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from collections import defaultdict
from .tokenizer.tokenizer import Tokenizer
from .vtree.v_tree import VTree
from .trie.trie import Trie, TrieNode
from statistics import mean
from math import log
import pickle

class NgramLanguageModel:
	@staticmethod
	def likelihoods_gen(TRAIN_CORPUS_DIR, TEST_CORPUS_DIR, N_GRAM=3):
		tokenized_train_corpus = Tokenizer.tokenizer(TRAIN_CORPUS_DIR)	

		# Creates two iterators: 
		# training_ngrams - sentences padded and turned into sequences of nltk.util.everygrams 
		# flat_sents - sentences padded as above and chained together for a flat stream of words
		training_ngrams, flat_sents = padded_everygram_pipeline(N_GRAM, tokenized_train_corpus)
		
		# Interpolated version of Kneser-Ney smoothing.
		lm = WittenBellInterpolated(N_GRAM)
		lm.fit(training_ngrams, flat_sents)
		UNK_SCORE = log(0.0002016942315449778)

		with open('bin/vocabulary.pkl', 'wb') as output:
			pickle.dump(lm.vocab, output)
			output.close()

		def build_v_tree():
		# Generate a Vocabulary Tree with 3 levels (trigram v-tree) containg likelihood estimate scores
		# The First Level will hold our entire vocabulary, along with their respective unigram scores (likelihoods)
		# Unkown word lookup in our current v-tree will return lm.score('<UNK>')
		# The Second Level will hold the bigram likelihoods that branch from their context word
		# If no word is found, the search will revert back to the upper (unigram) level
		# The Third Level will hold the trigram likelihoods that branch from their 2 context words
		# If no word is found, the search will revert back to the upper (bigram) level
		# If no word is found, the search will revert back to the even upper (unigram) level
			v_tree = VTree(UNK_SCORE)
			for sent in tokenized_train_corpus:
				for word in sent:
					v_tree.insert(target_word=word, lklhd=lm.logscore(word)) # 1st lvl - unigram score
				prev = '<s>'
				for word in sent:
					v_tree.insert(target_word=word, lklhd=lm.logscore(word, (prev,)), context=(prev,)) # 2nd lvl - bigram score
					prev = word
				prev_prev = '<s>'
				prev = '<s>'
				for word in sent:
					v_tree.insert(target_word=word, lklhd=lm.logscore(word, (prev_prev, prev)), context=(prev_prev, prev)) # 3rd level - trigram score
					prev_prev = prev
					prev = word
				with open('bin/v_tree.pkl', 'wb') as output:
					pickle.dump(v_tree, output)
					output.close()
			return v_tree
		v_tree = build_v_tree()

		def evaluate():
		# Evaluate the total entropy of the model with respect to the corpus.
		# This is the sum of the log probability of each word in the test corpus.
			tokenized_test_corpus = Tokenizer.tokenizer(TEST_CORPUS_DIR)
			test_ngrams, flat_sents = padded_everygram_pipeline(N_GRAM, tokenized_test_corpus)
			text_ngrams = [ngram for tokens in list(test_ngrams) for ngram in list(tokens)]
			entropy = 0
			for ngram in text_ngrams:
				word = ngram[-1]
				context = ngram[:-1]
				score = UNK_SCORE
				if not context:
					if lm.counts[word]:
						score = lm.logscore(word)
				elif lm.counts[context][word]:
					score = lm.logscore(word, context)
				entropy += score
			entropy *= -1/len(text_ngrams)
			file = open('bin/model_evaluation.txt', 'w')
			file.write('Model Evaluation Score (Entropy): {}\nModel Evaluation Score (Perplexity): {}'.format(entropy, pow(2.0, entropy)))
			file.close()
		evaluate()

def main():
	NgramLanguageModel.likelihoods_gen(TRAIN_CORPUS_DIR='../bin/train_corpus.pkl', \
									TEST_CORPUS_DIR='../bin/test_corpus.pkl')

if __name__ == "__main__":
   main()