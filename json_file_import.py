import pymysql
import json
import pandas as pd
import re
from nltk.corpus import stopwords
import string
import mysql.connector as sql
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# conn = pymysql.connect(
#     host='localhost',
#     port=3306,
#     user='root',
#     password='Database1',
#     db='yelp')
# cur = conn.cursor()
#
# # table business
# # create_table = 'CREATE TABLE business(business_id VARCHAR(100), name VARCHAR(300), address VARCHAR(2000), city VARCHAR (50), ' \
# #       'state VARCHAR(10), postal_code VARCHAR(100), latitude FLOAT, longitude FLOAT, stars FLOAT, review_count INT, ' \
# #                'is_open INT, attributes VARCHAR(2000), categories VARCHAR(2000), hours VARCHAR(500));'
#
# create_table = 'CREATE TABLE reviews(review_id VARCHAR(100), user_id VARCHAR(100), business_id VARCHAR(100), ' \
#       'stars FLOAT, useful INT, funny INT, cool INT, text TEXT, date DATETIME);'
# cur.execute(create_table)
# # import json into SQL
# with open('Yelp Dataset/yelp_academic_dataset_review.json') as file_review:
#     for i, line in enumerate(file_review):
#         dic = json.loads(line)
#         keys = ','.join(dic.keys())
#         valueList = [dici for dici in dic.values()]
#         for j, v in enumerate(valueList):
#             if isinstance(v, dict):
#                 valueList[j] = str(v)
#         valueTuple = tuple(valueList)
#         values = ','.join(['%s']*len(dic))
#         table = 'reviews'
#         sql = 'INSERT INTO {table}({keys}) VALUES ({values})'.format(table=table, keys=keys, values=values)
#         cur.execute(sql, valueTuple)
#         conn.commit()
# conn.close()

# select restaurants in Toronto
# query = 'SELECT * FROM business'
# cursor.execute(query)
# business = cursor.fetchall()
# df_business = pd.DataFrame(business)
# df_business.columns = ['business_id', 'name', 'address', 'city', 'state', 'postal_code', 'latitude',
#                        'longitude', 'stars', 'review_count', 'is_open', 'attributes', 'categories', 'hours']
# df_business = df_business[df_business['categories'].notna()]
#
# category_temp1 = ";".join(df_business['categories'])
# category_temp2 = re.split(";|,", category_temp1)
# # delete the left space (head space) of the item
# business_category_trim = [item.lstrip() for item in category_temp2]
# # delete the right space (end space) of the item
# business_category_trim = [item.rstrip() for item in business_category_trim]
# df_business_category = pd.DataFrame(business_category_trim, columns=['category'])
# restaurants = df_business.loc[[i for i in df_business['categories'].index if
#                                           re.search('Restaurants', df_business["categories"][i])]]
#
# # coordinates of Toronto
# lat_toronto = 43.6532
# lon_toronto = -79.3832
#
# lon_toronto_min, lon_toronto_max = lon_toronto - 0.5, lon_toronto + 0.5
# lat_toronto_min, lat_toronto_max = lat_toronto - 0.5, lat_toronto + 0.5
# restaurants_toronto = restaurants[(restaurants['longitude'] > lon_toronto_min) &
#                                                (restaurants['longitude'] < lon_toronto_max) &
#                                                (restaurants['latitude'] > lat_toronto_min) &
#                                                (restaurants['latitude'] < lat_toronto_max)]
'''
clean data by sql+pandas
'''
db = sql.connect(
    host='localhost',
    user='root',
    password='Database1',
    db='yelp')
# select restaurants in Toronto
cursor = db.cursor()
query = "SELECT * FROM business WHERE categories LIKE '%restaurant%' AND " \
         "latitude > 43.1532 AND latitude < 44.1532 AND longitude > -79.8832 AND longitude < -78.8832"
cursor.execute(query)
restaurants = cursor.fetchall()
# select reviews of restaurants in Toronto
query1 = "SELECT *" \
         "FROM reviews " \
         "WHERE business_id " \
         "IN (SELECT business_id FROM business WHERE categories LIKE '%restaurant%' AND " \
         "latitude > 43.1532 AND latitude < 44.1532 AND longitude > -79.8832 AND longitude < -78.8832)"
cursor.execute(query1)
reviews = cursor.fetchall()

df_restaurant = pd.DataFrame(restaurants)
df_restaurant.columns = ['business_id', 'name', 'address', 'city', 'state', 'postal_code', 'latitude',
                       'longitude', 'stars', 'review_count', 'is_open', 'attributes', 'categories', 'hours']
df_reviews = pd.DataFrame(reviews)
df_reviews.columns = ['review_id', 'user_id', 'business_id', 'stars', 'useful', 'funny', 'cool', 'text', 'date']

# turn a doc into clean tokens
def clean_review(text):
    text = re.sub('[^a-z\s]', '', text.lower())
    tokens = text.split()
# remove puntuations
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
# remove non-alphabetic
    tokens = [word for word in tokens if word.isalpha()]
# remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
# remove short tokens
    tokens = [word for word in tokens if len(word) > 1]
    stem = [stemmer.stem(w) for w in tokens]
    tokens = "".join(stem)
    lemma = [lemmatizer.lemmatize(o) for o in tokens]
    tokens = "".join(lemma)

    return tokens


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
df_reviews['text'] = df_reviews['text'].apply(clean_review)
from sqlalchemy import create_engine
df_reviews['text'] = df_reviews['text'].astype(str)
conn = create_engine('mysql+pymysql://root:Database1@localhost:3306/yelp', encoding='utf8')
df_reviews.to_sql('reviews_toronto', conn, if_exists='replace', index=False)