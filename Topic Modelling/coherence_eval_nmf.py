'''
Quora Questions
Evaluation of Topic Modeling NMF by using Coherence Score
Originally got about 0.39 coherence score for 20 topics following the same preprocessing as in Jupyter Notebook
I was able to improve the coherence score to 0.43 for 20 topics by implement an additional preprocessing step and tuning with max_df=0.95, min_df=5
Running time was ~60 seconds
'''

import pandas as pd
import re
import string
import nltk
import time
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

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
    
    # Track running time
    st = time.process_time()
    
    # Load the dataset (Replace with own path))
    file_path = "/Users/khantnyi/Desktop/NLP/week6/quora_questions.csv"
    df = pd.read_csv(file_path)

    # Preprocess the questions
    quora = df['Question'].dropna().tolist()
    preprocessed_quora = [preprocess_text(q) for q in quora]

    # Tfidf Vectorization with more aggressive filtering
    # Increase min_df to filter out rare words and keep max_df moderately low
    tfidf = TfidfVectorizer(max_df=0.95, min_df=5, stop_words='english')
    dtm = tfidf.fit_transform(preprocessed_quora)

    # Fit NMF Model
    nmf_model = NMF(n_components=20, random_state=42)
    nmf_model.fit(dtm)

    # Extract top words for each topic
    topics = []
    for index, topic in enumerate(nmf_model.components_):
        top_words = [tfidf.get_feature_names_out()[i] for i in topic.argsort()[-15:]]
        print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
        print(top_words)
        print('\n')
        topics.append(top_words)

    # Prepare texts for coherence model
    # We use the preprocessed texts here as well, since that aligns dictionary and topics
    texts = [doc.split() for doc in preprocessed_quora]

    # Create Gensim Dictionary and Corpus
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Calculate coherence score (c_v)
    coherence_model = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f"Coherence Score: {coherence_score}")
    
    et = time.process_time()
    print("Execution time:", et - st)

if __name__ == '__main__':
    main()
