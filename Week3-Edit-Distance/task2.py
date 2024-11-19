#Implement an algorithm to calculate the edit distance, given two words.

from nltk.corpus import words

def edit_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    
    #matrix to store distances between substrings
    dp = []
    for _ in range(m + 1): 
        row = []
        for _ in range(n + 1):
            row.append(0) 
        dp.append(row)

    #initialize the matrix and using algorithm
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,    # Deletion
                    dp[i][j - 1] + 1,    # Insertion
                    dp[i - 1][j - 1] + 1 # Substitution
                )

    return dp[m][n]

word1 = input('Enter the first word: ')
word2 = input('Enter the second word: ')
print(f'Edit distance between {word1} and {word2}: {edit_distance(word1, word2)}')
            
