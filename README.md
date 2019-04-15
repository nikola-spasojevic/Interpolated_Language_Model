# Interpolated_Language_Model
A language model based off KneserNeyInterpolated likelihood scores for trigrams, which are stored in 3 level deep vocabulary tree

In this iteration we decided to use a KneserNey Interpolation in order to distribute the score across trigrams/bigrams/unigrams 
using a set of lambdas which are optimised on the training set (lambda1 + lambda2 + lambda3 = 1).

Next is the use of a Back-Off model, where in which if the trigram score is below a certain threshold, the model will revert 
to the bigram model, and if this also falls below the threshold, the model reverts to the unigram model.

All model scores are stored within a vocabulary tree, which is represent as a tree of depth 3 (for trigram models).
With a vocabulary size of |V|, the size of the tree is calculated as |V|^N (N being the Ngram model in place).
