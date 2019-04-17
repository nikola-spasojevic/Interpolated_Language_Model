# Interpolated_Language_Model

A language model based off WittenBellInterpolated likelihood logscores for ngrams, which are stored in an N level deep vocabulary tree (N=3 for trigram models).

With a vocabulary size of |V|, the size of the tree is calculated as |V|<sup>N</sup>.

## Interpolation

Ngram probability mass is distributed across lower level ngrams (i.e. likelihood estimates from trigram scores are weighed down and given to bigram/unigram scores given a set of weights that add up to 1 (&#955;1 + &#955;2 + &#955;3 = 1) ). These weights are optimised using the test set.

P<sub>ip</sub>(W<sub>n</sub> | W<sub>n-2</sub>, W<sub>n-1</sub>) = &#955;1 P(W<sub>n</sub> | W<sub>n-2</sub>, W<sub>n-1</sub>) +
&#955;2 P(W<sub>n</sub> | W<sub>n-1</sub>) + &#955;3 P(W<sub>n</sub>)

## Back-Off models

Next is the use of a Back-Off model, where in which if the trigram score is below a certain threshold, the model will revert 
to the bigram model, and if this also falls below the threshold, the model reverts to the unigram model.

## Evaluation

The scores from the test set are calculated in a log space in order to avoid underflow (probabilities of some sequences are very small) and to have more efficient evaluation calculations - logscore addition is faster than score multiplcation.

log(p1 x p2 x p3 x p4 ) = log p1 + log p2 + log p3 + log p4

1. We train parameters of our model on a training set.

2. We test the model’s performance on data we haven’t seen. A test set is an unseen dataset that is different from our training set, totally unused.

3. An evalua1on metric tells us how well our model does on the test set.

### Perplexity

Perplexity is the inverse probability of the test set, normalized by the number of words:

PP(W) = P(w1w2w3...wN)<sup>-1/N</sup>

Minimizing perplexity is the same as maximizing probability.

### Overfitting

Ngrams only work well for word prediction if the test corpus looks like the training corpus. 

Hence, we need to have a more general model that can account for the zero counts appearing from the test set for unseen words
