'''
Yelp dataset
Data preparation
Train Embedding layer
Train word2vec embedding
Use pre-trained embedding
Naive Bayes
'''


import pandas as pd
import re
import mysql.connector as sql
import numpy as np
from nltk.corpus import stopwords
import string
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
import keras
from keras import layers
from tqdm import tqdm
from sklearn.model_selection import KFold


lemmatizer = WordNetLemmatizer()
db = sql.connect(
    host='localhost',
    user='root',
    password='Database1',
    db='yelp')
cursor = db.cursor()

# load the cleaned reviews data
query1 = "SELECT *" \
         "FROM reviews " \
         "WHERE business_id " \
         "IN (SELECT business_id FROM business WHERE categories LIKE '%restaurant%' AND " \
         "latitude > 43.1532 AND latitude < 44.1532 AND longitude > -79.8832 AND longitude < -78.8832 AND review_count>100)"
cursor.execute(query1)
reviews = cursor.fetchall()
print('Data loaded')

df_reviews = pd.DataFrame(reviews)
df_reviews.columns = ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'text', 'date']
# 1: pos, 0: neg
df_reviews['label'] = ''
df_reviews['label'].loc[(df_reviews['stars'] >= 4.0)] = 1
df_reviews['label'].loc[(df_reviews['stars'] < 4.0)] = 0


def clean_review(text):
    text = re.sub('[^a-z\s]', '', text.lower())
    tokens = text.split()
    # remove punctuations
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove non-alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # remove short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


# save tokens
def save_file(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def load_file(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# remove words in the reviews that only appeared once
def remove_redundant(text):
    words = []
    t = [w for w in text if w in vocab]
    for w in t:
        words.append(lemmatizer.lemmatize(w))
    # # for naive bayes
    # words = " ".join(words)
    return words


# load pre-trained embedding model
def load_embedding(filename):
    file = open(filename, 'r')
    lines = file.readlines()[1:]
    file.close()
    embedding = {}
    for line in lines:
        parts = line.split()
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return embedding


def get_weight_matrix(embedding, vocab):
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, 100))
    for word, i in vocab.items():
        weight_matrix[i] = embedding.get(word)
    return weight_matrix


vocab = Counter()
print('Cleaning data...')
# convert words into tokens
df_reviews['text'] = df_reviews['text'].apply(clean_review)

for i in tqdm(df_reviews['text']):
        vocab.update(i)

min_occurrence = 3
# only consider words appeared more than 3 times
tokens = [k for k, c in vocab.items() if c >= min_occurrence]

save_file(tokens, 'vocab.txt')
vocab = load_file('vocab.txt')
vocab = vocab.split()
vocab = set(vocab)

print("Cleaning data...")
df_reviews['text'] = df_reviews['text'].apply(remove_redundant)
# prepare training set, select 60000 samples to train the model
index = np.random.randint(len(df_reviews.index), size=60000)
x_train = df_reviews['text'][index]
x_train.reset_index(drop=True, inplace=True)
y_train = df_reviews['label'][index]
y_train.reset_index(drop=True, inplace=True)
x_train.to_pickle('train data')

# k-fold cross validation
kfold = KFold(n_splits=6, shuffle=True)
fold_no = 1
acc_fold = []
loss_fold = []
max_length = max([len(s) for s in x_train])
y_train = y_train.astype(np.float32)
for train, test in kfold.split(x_train, y_train):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)
    encoded_train = tokenizer.texts_to_sequences(x_train[train])
    encoded_test = tokenizer.texts_to_sequences(x_train[test])
    Xtrain = pad_sequences(encoded_train, maxlen=max_length, padding='post')
    Xtest = pad_sequences(encoded_test, maxlen=max_length, padding='post')

    vocab_size = len(tokenizer.word_index) + 1

# CNN model
    model = keras.Sequential()
    # words will be embedded into 100-long vector (output)
    # turn positive integers (indexes) into dense vectors of fixed size
    model.add(layers.Embedding(vocab_size, 100, input_length=max_length))
    model.add(layers.Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print(model.summary())

    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('-------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    # history = model.fit(Xtrain, y_train, epochs=10, verbose=2)
    history = model.fit(Xtrain, y_train[train], epochs=5, verbose=2)
    # loss, acc = model.evaluate(Xtest, y_test, verbose=0)
    loss, acc = model.evaluate(Xtest, y_train[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {loss}; {model.metrics_names[1]} of {acc*100}%')
    acc_fold.append(acc*100)
    loss_fold.append(loss)
    fold_no += 1

# model.save('filename')