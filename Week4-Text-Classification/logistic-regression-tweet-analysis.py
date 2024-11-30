# Q) Given a dataset, implement a sentiment classifier. You are free to choose any pre-processing, feature extraction and classifier model. 
# Evaluate your model in terms of precision, recall and F1 score.

# Install Necessary Libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np


# Step 1: Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Step 2: Loading the dataset (replace with your path)
file_path = 'Tweets.csv'
tweets_df = pd.read_csv(file_path)

# Step 3: Defining target and text columns from the dataset
target_column = 'airline_sentiment'
text_column = 'text'

# Step 4: Preprocessing the text data
tweets_df['preprocessed_text'] = tweets_df[text_column].apply(preprocess_text)

# Step 5: Defining features and labels from csv
X = tweets_df['preprocessed_text']
y = tweets_df[target_column]

# Step 6: Trying different train/test ratios
split_ratios = [(0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]

for train_size, test_size in split_ratios:
    print(f"\n--- Split Ratio: Train {train_size*100}%, Test {test_size*100}% ---")
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=train_size, 
        stratify=y, 
        random_state=42
    )

    # Step 7: Vectorizing the text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Step 8: Applying Logistic Regression Classifier
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)

    # Predictions
    y_pred = lr_model.predict(X_test_tfidf)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n")
    print(cm)

    # Confusion Matrix Labels
    labels = sorted(y.unique())
    print(f"\nConfusion Matrix Labels: {labels}")

