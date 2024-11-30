# Q) Construct a Naive Bayes classifier for movie reviews.
# Try different train/test splitting ratios. eg. 70:30, 80:20 and 90:10 
# Evaluate the models (precision, recall, f1 score). Generate the confusion matrix. Record your findings, analyze and compare the model performance in your own words.

# Install Necessary Libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize text
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join words back into a single string
    return " ".join(words)

# Read and Preprocess Reviews
def load_and_preprocess_reviews(file_path):
    reviews = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if '\t' in line:
                try:
                    review, label = line.strip().split('\t')
                    # Preprocess the review
                    review = preprocess_text(review)
                    reviews.append(review)
                    labels.append(int(label))
                except ValueError:
                    continue
    return pd.DataFrame({"review": reviews, "label": labels})

# Load the dataset (replace with your own path)
df = load_and_preprocess_reviews("movie_reviews.txt")

# Define train/test split ratios
split_ratios = [(0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]

results = []

for train_size, test_size in split_ratios:
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["label"], train_size=train_size, stratify=df["label"], random_state=42
    )
    
    # Vectorize the text data
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train the Naive Bayes classifier
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_vec)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    
    # Record results
    results.append({
        "Train/Test Split": f"{int(train_size*100)}/{int(test_size*100)}",
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Confusion Matrix": conf_matrix
    })

# Display results
for result in results:
    print(f"Train/Test Split: {result['Train/Test Split']}")
    print(f"Precision: {result['Precision']:.2f}")
    print(f"Recall: {result['Recall']:.2f}")
    print(f"F1 Score: {result['F1 Score']:.2f}")
    print("Confusion Matrix:")
    print(result["Confusion Matrix"])
    print("-" * 50)


