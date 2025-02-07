import pickle
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Concatenate, Dropout, Permute, Activation, add, dot

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to normalize text (tense handling and punctuation cleanup)
def preprocess_text(text):
    text = re.sub(r'\s*([?.!,])\s*', r'\1 ', text)
    words = text.lower().split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Load training and test data
with open("train_qa.dat", "rb") as fp:
    train_data = pickle.load(fp)

with open("test_qa.dat", "rb") as fp:
    test_data = pickle.load(fp)

# Create a set that holds the vocab words
vocab = set()
all_data = test_data + train_data
for story, question, answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))
vocab.add('no')
vocab.add('yes')

# Tokenizer
tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)
vocab_size = len(tokenizer.word_index) + 1

# Save tokenizer
with open("tokenizer.pkl", "wb") as fp:
    pickle.dump(tokenizer, fp)

# Determine max story and question lengths
max_story_len = max(len(story) for story, _, _ in train_data)
max_question_len = max(len(query) for _, query, _ in train_data)

with open("config.pkl", "wb") as fp:
    pickle.dump((max_story_len, max_question_len), fp)

# Vectorization function
def vectorize_stories(data, word_index, max_story_len, max_question_len):
    X, Xq, Y = [], [], []
    for story, query, answer in data:
        story = preprocess_text(" ".join(story)).split()
        query = preprocess_text(" ".join(query)).split()
        x = [word_index.get(word, 0) for word in story]
        xq = [word_index.get(word, 0) for word in query]
        y = np.zeros(len(word_index) + 1)
        if answer in word_index:
            y[word_index[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (
        pad_sequences(X, maxlen=max_story_len), 
        pad_sequences(Xq, maxlen=max_question_len), 
        np.array(Y)
    )

# Vectorize datasets
X_train, Xq_train, Y_train = vectorize_stories(train_data, tokenizer.word_index, max_story_len, max_question_len)
X_test, Xq_test, Y_test = vectorize_stories(test_data, tokenizer.word_index, max_story_len, max_question_len)

# Model definition
input_sequence = Input((max_story_len,))
question_input = Input((max_question_len,))
input_encoder_m = Embedding(input_dim=vocab_size, output_dim=64)(input_sequence)
input_encoder_m = Dropout(0.3)(input_encoder_m)
input_encoder_c = Embedding(input_dim=vocab_size, output_dim=max_question_len)(input_sequence)
input_encoder_c = Dropout(0.3)(input_encoder_c)
question_encoder = Embedding(input_dim=vocab_size, output_dim=64, input_length=max_question_len)(question_input)
question_encoder = Dropout(0.3)(question_encoder)
input_encoded_m = input_encoder_m
input_encoded_c = input_encoder_c
question_encoded = question_encoder
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)
response = add([match, input_encoded_c])
response = Permute((2, 1))(response)
answer = Concatenate()([response, question_encoded])
answer = LSTM(32)(answer)
answer = Dropout(0.5)(answer)
answer = Dense(vocab_size, activation='softmax')(answer)
model = Model([input_sequence, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit([X_train, Xq_train], Y_train, batch_size=32, epochs=120, validation_data=([X_test, Xq_test], Y_test))
model.save("chatbot_model.h5")

# Evaluation
model = load_model("chatbot_model.h5")
with open("tokenizer.pkl", "rb") as fp:
    tokenizer = pickle.load(fp)
with open("config.pkl", "rb") as fp:
    max_story_len, max_question_len = pickle.load(fp)
X_test, Xq_test, Y_test = vectorize_stories(test_data, tokenizer.word_index, max_story_len, max_question_len)
pred_results = model.predict([X_test, Xq_test])
predicted_answers = np.argmax(pred_results, axis=1)
true_answers = np.argmax(Y_test, axis=1)

# Display sample result
test_index = 0
story = ' '.join(test_data[test_index][0])
query = ' '.join(test_data[test_index][1])
true_answer = test_data[test_index][2]
predicted_word = [word for word, index in tokenizer.word_index.items() if index == predicted_answers[test_index]]
predicted_word = predicted_word[0] if predicted_word else "Unknown"
print("\nStory:\n", story)
print("\nQuery:\n", query)
print("\nTrue Test Answer:", true_answer)
print("\nPredicted Answer:", predicted_word)

# Accuracy
accuracy = np.mean(predicted_answers == true_answers)
print(f"\nModel Accuracy: {accuracy:.4f}")
