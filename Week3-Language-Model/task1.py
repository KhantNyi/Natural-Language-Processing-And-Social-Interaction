# Task 1: Build a unigram language model based on the Reuters corpus from NLTK. Generate 20-word sequences using the unigram model constructed. Since there is no order dependency in a unigram model, print 20 words with highest probabilities.

from nltk.corpus import reuters
from collections import Counter

# Tokenizing the words
words = [word.lower() for word in reuters.words()]

# Counting word frequencies
word_counts = Counter(words)
total_words = sum(word_counts.values())

# Calculating probabilities for each word
unigram_probabilities = {word: count / total_words for word, count in word_counts.items()}

top_20_words = sorted(unigram_probabilities.items(), key=lambda x: x[1], reverse=True)[:20]

print(" ".join([word for word, _ in top_20_words]))
