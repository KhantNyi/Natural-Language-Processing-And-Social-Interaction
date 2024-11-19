# Implement a program that reads a text file and locates all words that are misspelled (not in the dictionary).The program must provide suggestions for each misspelled word based on the edit distance.If there are multiple suggestions, which word would you recommend?

# I would recommend words with mininum edit distance (already implemented from other tasks) and most frequently used words. I will use brown corpus to filter the most frequently used words.

from nltk.corpus import words, brown
from nltk.tokenize import word_tokenize
from collections import Counter

dictionary = set(words.words())
word_frequencies = Counter(brown.words())

def edit_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    
    # Initialize the matrix
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize the first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,   # Deletion
                    dp[i][j - 1] + 1,   # Insertion
                    dp[i - 1][j - 1] + 1  # Substitution
                )

    return dp[m][n]

def get_best_suggestion(misspelled_word):
    
    if misspelled_word.lower() not in dictionary:
        suggestions = [
            (edit_distance(misspelled_word.lower(), word.lower()), word)
            for word in dictionary
        ]
        suggestions.sort()

        # Filter the suggestions with the smallest edit distance
        min_distance = suggestions[0][0]
        closest_words = [word for dist, word in suggestions if dist == min_distance]

        # Choose the most frequent word among the closest words
        best_suggestion = max(closest_words, key=lambda w: word_frequencies[w.lower()])
        return best_suggestion

def spell_check(filename):
    with open(filename, 'r') as file:
        text = file.read()

    words_in_text = word_tokenize(text)

    misspelled_words = {}
    for word in words_in_text:
            if word not in misspelled_words:
                best_suggestion = get_best_suggestion(word)
                misspelled_words[word] = best_suggestion

    return misspelled_words

filename = 'week3/check_me.txt'  # Replace with your text file path
misspelled_words = spell_check(filename)
for word, suggestion in misspelled_words.items():
    print(f"'{word}' is misspelled. Best suggestion: '{suggestion}'")
    
# Note: It takes about 30 second to run a text file with 8 mispelled words so if it looks stuck, do not panic.
# The code does run, it is just (very) slow.
