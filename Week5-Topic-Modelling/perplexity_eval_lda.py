'''
Evaluation of Topic Modeling LDA using perplexity
4550 perplexity score for 20 topics with Count Vectorization method that was used in Jupyter Notebook
I was able to improve the perplexity score to 2878 for the same 20 topics (2888 for 10 topics therefore marginal difference) by adding a more detailed preprocessing step and tuning with max_df=0.95, min_df=5
Running time was ~200 seconds due to large amount of data in csv and LDA processing
'''

import time
import pandas as pd
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Ensure that NLTK resources are downloaded (if not already)
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)

def preprocess_text(text, min_len=2):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Split into tokens
    tokens = text.split()
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    # Remove short tokens
    tokens = [t for t in tokens if len(t) >= min_len]
    return " ".join(tokens)

def main():
    st = time.process_time()
    
    # Load the dataset
    file_path = "/Users/khantnyi/Desktop/NLP/week6/quora_questions.csv"
    df = pd.read_csv(file_path)

    # Preprocess the questions
    quora = df['Question'].dropna().tolist()
    preprocessed_quora = [preprocess_text(q) for q in quora]

    # Count Vectorization with more aggressive filtering
    # Increase min_df to filter out rare words and keep max_df moderately low
    cv = CountVectorizer(max_df=0.95, min_df=5, stop_words='english')
    dtm = cv.fit_transform(preprocessed_quora)

    # Fit LDA Model
    lda_model = LatentDirichletAllocation(n_components=20, random_state=42)
    lda_model.fit(dtm)

    # Extract top words for each topic
    for index, topic in enumerate(lda_model.components_):
        top_words = [cv.get_feature_names_out()[i] for i in topic.argsort()[-15:]]
        print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
        print(top_words)
        print('\n')

    # Calculate and display perplexity
    perplexity = lda_model.perplexity(dtm)
    print(f"Perplexity Score: {perplexity}")
    
    et = time.process_time()
    print(f"Time taken: {et-st} seconds")

if __name__ == '__main__':
    main()
