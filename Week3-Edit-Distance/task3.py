#Implement a program that suggets a word with correct spelling based on English dictionary (given a misspelled word).

from nltk.corpus import words

def edit_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i  
    for j in range(n + 1):
        dp[0][j] = j 
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if word1[i - 1].lower() == word2[j - 1].lower() else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )
    return dp[m][n]

def suggest_correct_spelling(word):
    word_list = words.words()
    word_set = set(w.lower() for w in word_list)
    
    # checking if it's correctly spelled
    if word.lower() in word_set:
        return [word]  
    
    candidates = []
    for candidate in word_list:
        candidate_lower = candidate.lower()
        
        # Skip candidates with a large length difference
        if abs(len(candidate_lower) - len(word)) > 2:
            continue
        
        distance = edit_distance(word.lower(), candidate_lower)
        candidates.append((candidate, distance))
        
    # Sorting candidates by edit distance
    candidates.sort(key=lambda x: x[1])
    
    # Returning the top 3 suggestions
    suggestions = []
    for candidate, _ in candidates[:3]:
        suggestions.append(candidate)
    return suggestions

word = input('Enter a word to check spelling: ')
top_three = suggest_correct_spelling(word)
print(f'Suggested corrections: {top_three}')

