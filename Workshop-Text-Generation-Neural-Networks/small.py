import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Input, Dropout
from keras.utils import to_categorical
import re
from collections import Counter

# Load books
books = [
    r'\Harry_Potter_1_Sorcerers_Stone.txt',
    r'\Harry_Potter_2-The_Chamber_of_Secrets.txt',
    r'\Harry_Potter_3_Prisoner_of_Azkaban.txt',
    r'\Harry_Potter_4_The_Goblet_of_Fire.txt'
]

# Normalize function
def normalize(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Combine and normalize text
combined_text = '\n'.join([open(book, 'r', encoding='utf-8', errors='ignore').read() for book in books])
normalized_text = normalize(combined_text)

# Tokenize and filter vocabulary
words = normalized_text.split()
word_freq = Counter(words)
most_common_words = {word for word, _ in word_freq.most_common(5000)}  # Top 5000 words
if len(word_freq) < 5000:  # Handle small datasets
    most_common_words = {word for word, _ in word_freq.most_common(len(word_freq))}

words = [word for word in words if word in most_common_words]
word_list = list(most_common_words)
word_indices = {word: idx for idx, word in enumerate(word_list)}

# Prepare sequences
sequence_length = 5
X, Y = [], []
for i in range(len(words) - sequence_length):
    X.append([word_indices[words[j]] for j in range(i, i + sequence_length)])
    Y.append(word_indices[words[i + sequence_length]])

X = np.array(X).reshape(-1, sequence_length, 1)  # Reshape for LSTM
Y = np.array(Y)

# Build the model
model_small = Sequential()
model_small.add(Input(shape=(sequence_length, 1))) 
model_small.add(LSTM(64, return_sequences=False))
model_small.add(Dropout(0.2))  # Regularization
model_small.add(Dense(len(word_list), activation='softmax'))
model_small.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Train the model
model_small.fit(X, Y, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model_small.save('text_gen_model.h5')

# Generate text
def generate_text(model, word_list, word_indices, num_words=20, sequence_length=5):
    start_word = np.random.choice(word_list)
    current_words = [word_indices[start_word]]
    generated_text = [start_word]

    for _ in range(num_words):
        if len(current_words) < sequence_length:
            padded_sequence = [0] * (sequence_length - len(current_words)) + current_words
        else:
            padded_sequence = current_words[-sequence_length:]

        input_sequence = np.array(padded_sequence).reshape(1, sequence_length, 1)
        prediction = model.predict(input_sequence, verbose=0).flatten()
        next_word_index = np.random.choice(len(word_list), p=prediction)  # Sample with randomness
        next_word = word_list[next_word_index]
        generated_text.append(next_word)
        current_words.append(next_word_index)

    return ' '.join(generated_text)

# Generate and print text
generated_text = generate_text(model_small, word_list, word_indices, num_words=50)
print("Generated Text:\n", generated_text)

# Save generated text to a file
with open('generated_chapter_small.txt', 'w') as f:
    f.write(generated_text)
