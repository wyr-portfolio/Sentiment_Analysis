import pandas as pd
import re
import mysql.connector as sql
import numpy as np
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.metrics import accuracy_score
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
    words = []
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
    # remove redundancy
    t = [w for w in tokens if w in vocab]
    for w in t:
        words.append(lemmatizer.lemmatize(w))
    words = " ".join(words)
    return words


def load_file(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# save_file(tokens, 'vocab.txt')
vocab = load_file('vocab.txt')
vocab = vocab.split()
vocab = set(vocab)

pos = []
neg = []
count_pos = {}
count_neg = {}

x_train, x_test, y_train, y_test = train_test_split(df_reviews['text'], df_reviews['label'], test_size=0.15)
x_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
x_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
for i, text in enumerate(x_train):
    text = clean_review(text).split(" ")
    if y_train[i] == 0:
        for t in text:
            neg.append(t)
            count_neg.setdefault(t, 0)
            count_neg[t] += 1
    else:
        for t in text:
            pos.append(t)
            count_pos.setdefault(t, 0)
            count_pos[t] += 1
unique_words_neg = list(set(neg))
unique_words_pos = list(set(pos))

correct = 0
wrong = 0
for j, review in enumerate(x_test):
    y_guess = -1
    review = clean_review(review).split(" ")
    possibility_neg = 0
    possibility_pos = 0
    for r in review:
        if r in count_neg:
            possibility_neg += np.log((count_neg[r] + 1) / (len(neg) + len(unique_words_neg) + len(unique_words_pos)))
            possibility_pos += np.log(1 / (len(pos) + len(unique_words_neg) + len(unique_words_pos)))
        elif r in count_pos:
            possibility_neg += np.log(1 / (len(neg) + len(unique_words_neg) + len(unique_words_pos)))
            possibility_pos += np.log((count_pos[r] + 1) / (len(pos) + len(unique_words_pos) + len(unique_words_neg)))
    possibility_neg += np.log(len(neg) / (len(neg) + len(pos)))
    possibility_pos += np.log(len(pos) / (len(neg) + len(pos)))
    if possibility_neg > possibility_pos:
        y_guess = 1
    elif possibility_pos > possibility_neg:
        y_guess = 0
    if y_guess == y_test[j]:
        correct += 1
    elif y_guess != y_test[j] and y_guess != -1:
        wrong += 1
score = correct / len(x_test)
loss = wrong / len(x_test)
print(f'Score for {score * 100}%')
print(f'Loss for {loss * 100}%')


################################ multinomialNB ##########################################################################
# index = np.random.randint(len(df_reviews.index), size=400)
# x = df_reviews['text'][index]
# x.reset_index(drop=True, inplace=True)
# y = df_reviews['label'][index]
# y.reset_index(drop=True, inplace=True)
# # k-fold cross validation
# kfold = KFold(n_splits=10, shuffle=True)
# fold_no = 1
# acc_fold = []
# loss_fold = []
# model_nb = mnb()

# for train, test in kfold.split(x, y):
    # max_length = max([len(s) for s in x[train]])
    # vectorizer = CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words=None, max_features=max_length)
    # train_data_features = vectorizer.fit_transform(x[train])
    # train_data_features = train_data_features.toarray()
    # test_data_features = vectorizer.fit_transform(x[test])
    # test_data_features = test_data_features.toarray()
    #
    # print('-------------------------------------------------------------------------')
    # print(f'Training for fold {fold_no} ...')
    # train_y = y[train].astype(np.float32)
    # test_y = y[test].astype(np.float32)
    # history = model_nb.fit(train_data_features, train_y)
    # # score = np.mean(cross_val_score(model_nb, train_data_features, y, cv=10, scoring='roc_auc'))
    #
    # result = model_nb.predict(test_data_features)
    # accuracy = accuracy_score(test_y, result)
    # print(f'Score for fold {fold_no}: {accuracy*100}%')
    # acc_fold.append(accuracy*100)
    # fold_no += 1
########################################################################################################################