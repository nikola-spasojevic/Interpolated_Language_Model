# Interpolated_Language_Model

A language model based off WittenBellInterpolated likelihood logscores for ngrams, which are stored in an N level deep vocabulary tree (N=3 for trigram models).

With a vocabulary size of |V|, the size of the tree is calculated as |V|<sup>N</sup>.

## Interpolation

Ngram probability mass is distributed across lower level ngrams (i.e. likelihood estimates from trigram scores are weighed down and given to bigram/unigram scores given a set of weights that add up to 1 (&#955; lambda1 + lambda2 + lambda3 = 1)). These weights are optimised using the test set.

## Back-Off models

Next is the use of a Back-Off model, where in which if the trigram score is below a certain threshold, the model will revert 
to the bigram model, and if this also falls below the threshold, the model reverts to the unigram model.


