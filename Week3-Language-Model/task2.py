# Task 2: Build a bigram language model based on the reuters corpus from NLTK. Generate a 20-word sequence using the bigram model constructed. Randomize the first word.

import random
from nltk.corpus import reuters

# Preprocessing
def preprocess(corpus):
    words = [word.lower() for word in corpus]
    return words

words = preprocess(reuters.words())

# Creating bigrams
def create_bigrams(words):
    bigrams = []
    for i in range(len(words) - 1):
        bigrams.append((words[i], words[i + 1]))
    return bigrams

bigram_tuples = create_bigrams(words)

# Checking frequency of bigrams
def frequency_check(bigrams):
    cfd = {}
    for word1, word2 in bigrams:
        if word1 not in cfd:
            cfd[word1] = {}
        if word2 not in cfd[word1]:
            cfd[word1][word2] = 0
        cfd[word1][word2] += 1 # counting occurrences of word2 after word1
    return cfd

cfd = frequency_check(bigram_tuples)

# Text generation
def generate_text(start_word, length=20):
    result = [start_word]
    current_word = start_word
    for _ in range(length - 1):
        if current_word not in cfd:
            break
        possible_next_words = list(cfd[current_word].keys())
        weights = [cfd[current_word][word] for word in possible_next_words]
        next_word = random.choices(possible_next_words, weights=weights)[0]
        
        result.append(next_word)
        current_word = next_word
    
    return ' '.join(result)

# Randomizing the first word from the corpus
first_word = random.choice(words)

# Generating a 20-word sequence
generated_sequence = generate_text(first_word)
print(generated_sequence)
