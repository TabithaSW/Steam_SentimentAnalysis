# nltk model, we may try some scipy or other ml libraries. this is the initial model.


"""
naive Bayes classifiers are a family of linear "probabilistic classifiers". It is commonly used for text classification tasks like sentiment analysis.
meaning they assume that the features are conditionally independent, given the target class. hence the word, naive.
conditional independence describes situations wherein an observation is irrelevant or redundant when evaluating the certainty of a hypothesis.
steam reviews can be considered independent, however not incredibly strictly due to the context, such as not good or very good, for example, as strings.
more complex sentiment analysis tasks where word dependencies and context are critical, other methods like deep learning models (e.g., Recurrent Neural Networks or Transformers) may be more appropriate.
this will be the initial model, since naive bayes works well for large sets and multi-class problems, and requires less trianing data.
"""

import pandas
import numpy
import nltk

# Naive Bayes calculates the probability of a review belonging to a  class based on the probabilities of individual words occurring in that class.