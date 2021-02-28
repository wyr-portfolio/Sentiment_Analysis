import pandas as pd
import numpy as np
import json
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from nltk.corpus import stopwords
from wordcloud import WordCloud


business = []
tips = []
with open('Yelp Dataset/yelp_academic_dataset_business.json') as file_business:
    for i, line in enumerate(file_business):
        business.append(json.loads(line))
df_business = pd.DataFrame(business)
df_business = df_business[df_business['categories'].notna()]

category_temp1 = ";".join(df_business['categories'])
category_temp2 = re.split(";|,", category_temp1)
# delete the left space (head space) of the item
business_category_trim = [item.lstrip() for item in category_temp2]
# delete the right space (end space) of the item
business_category_trim = [item.rstrip() for item in business_category_trim]
df_business_category = pd.DataFrame(business_category_trim, columns=['category'])


# plot top 10 categories
business_category_count = df_business_category.category.value_counts()
business_category_count = business_category_count.sort_values(ascending=False)
business_category_count = business_category_count.iloc[0:10]

# fig = plt.figure(figsize=(10, 6))
# ax = sns.barplot(business_category_count.index, business_category_count.values)
# plt.title("Top Business Categories", fontsize=10)
# x_locs, x_labels = plt.xticks()
# plt.setp(x_labels, rotation=45)
# plt.ylabel('Number of Businesses', fontsize=12)
# plt.xlabel('Category', fontsize=12)
#
# # plot geographical visualization
# r = ax.patches
# labels = business_category_count.values
# for rect, label in zip(r, labels):
#     height = rect.get_height()
#     ax.text(rect.get_x() + rect.get_width()/2, height+10, label, ha='center', va='bottom')
#
df_business_restaurant = df_business.loc[[i for i in df_business['categories'].index if
                                          re.search('Restaurants', df_business["categories"][i])]]
# fig = plt.figure(figsize=(10, 6))
# plt.title('Geographic View of Restaurant Locations', fontsize=20)
#
# # choose the type of the map
# ax_map = plt.axes(projection=ccrs.PlateCarree())
# # change to underlay image by setting to ax.stock_img(); add coastline to the map by ax.coastlines()
# ax_map.stock_img()
# ax_map.add_feature(cfeature.BORDERS, edgecolor='gray')
# # zoom in North American region
# ax_map.set_extent([-150, -50, 10, 60])
# coord = (df_business_restaurant['longitude'].tolist(), df_business_restaurant['latitude'].tolist())
# plt.scatter(coord[0], coord[1], s=2, c='red', linewidths=3, transform=ccrs.PlateCarree())
# plt.show()
#
# plot top 10 cities with highest number of restaurants
df_business_restaurant['city_state'] = df_business_restaurant['city'] + ',' + df_business_restaurant['state']
city_restaurant_count = df_business_restaurant.city_state.value_counts()
city_restaurant_count = city_restaurant_count.sort_values(ascending=False)
city_restaurant_count = city_restaurant_count.iloc[0:10]

# fig_restaurant = plt.figure(figsize=(10, 6))
# ax_restaurant = sns.barplot(city_restaurant_count.index, city_restaurant_count.values)
# plt.title("Top cities", fontsize=20)
# x_rest, x_labels_rest = plt.xticks()
# plt.setp(x_labels_rest, rotation=60)
# plt.xlabel('City, State', fontsize=12)
# plt.ylabel('Number of Restaurants', fontsize=12)
# labels_restaurant = city_restaurant_count.values
# for rect_rest, label_rest in zip(r, labels_restaurant):
#     height_rest = rect_rest.get_height()
#     ax_restaurant.text(rect_rest.get_x()+rect_rest.get_width()/2, height_rest+10, label_rest, ha='center', va='bottom')
#
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

# # coordinates of Toronto
# lat_toronto = 43.6532
# lon_toronto = -79.3832
#
# lon_toronto_min, lon_toronto_max = lon_toronto - 0.5, lon_toronto + 0.5
# lat_toronto_min, lat_toronto_max = lat_toronto - 0.5, lat_toronto + 0.5
# df_restaurant_toronto = df_business_restaurant[(df_business_restaurant['longitude'] > lon_toronto_min) &
#                                                (df_business_restaurant['longitude'] < lon_toronto_max) &
#                                                (df_business_restaurant['latitude'] > lat_toronto_min) &
#                                                (df_business_restaurant['latitude'] < lat_toronto_max)]
# df_restaurant_toronto.plot(kind='scatter', x='longitude', y='latitude',
#                            color='#52fff3', s=0.02, alpha=0.6, subplots=True, ax=ax1)
# ax1.set_title('Restaurant in Toronto')
# ax1.set_facecolor('black')
#
# # coordinates of Las Vegas
# lat_vegas = 36.1699
# lon_vegas = -115.1398
# lon_vegas_min, lon_vegas_max = lon_vegas - 0.5, lon_vegas + 0.5
# lat_vegas_min, lat_vegas_max = lat_vegas - 0.5, lat_vegas + 0.5
# de_restaurant_vegas = df_business_restaurant[(df_business_restaurant['longitude'] > lon_vegas_min) &
#                                                (df_business_restaurant['longitude'] < lon_vegas_max) &
#                                                (df_business_restaurant['latitude'] > lat_vegas_min) &
#                                                (df_business_restaurant['latitude'] < lat_vegas_max)]
# de_restaurant_vegas.plot(kind='scatter', x='longitude', y='latitude',
#                            color='#52fff3', s=0.02, alpha=0.6, subplots=True, ax=ax2)
# ax2.set_title('Restaurant in Las Vegas')
# ax2.set_facecolor('black')
#
# f.tight_layout(pad=5.0)

# # Top common restaurants
# df_business_restaurant.loc[df_business_restaurant.name == 'Subway', 'name'] = 'Subway Restaurants'
# restaurant_count = df_business_restaurant.name.value_counts()
# restaurant_count = restaurant_count.sort_values(ascending=False)
# restaurant_count = restaurant_count.iloc[0:15]
#
# fig = plt.figure(figsize=(10, 6))
# ax = sns.barplot(restaurant_count.index, restaurant_count.values)
# plt.title('Restaurant with High Occurrence', fontsize=20)
# x_locs, x_labels = plt.xticks()
# plt.setp(x_labels, rotation=60)
# plt.ylabel('Number of Restaurants', fontsize=12)
# plt.xlabel('Restaurant', fontsize=12)
# r = ax.patches
# labels = restaurant_count.values
# for rect, label in zip(r, labels):
#     height = rect.get_height()
#     ax.text(rect.get_x() + rect.get_width()/2, height+10, label, ha='center', va='bottom')

# # Median of Chipotle
# Chipotle_med = df_business_restaurant.loc[df_business_restaurant.name=='Chipotle Mexican Grill'].stars.median()
# fig = plt.figure(figsize=(8, 6))
# sns.scatterplot(x='stars', y='review_count', data=df_business_restaurant)
# plt.title('Reviews vs Rating', fontsize=20)
# plt.xlabel('Rating', fontsize=12)
# plt.ylabel('Reviews', fontsize=12)


# replace None in attributes to {}
df_business_restaurant['attributes'] = df_business_restaurant['attributes'].apply(lambda x: {} if x is None else x)
# parsing attributes
df_attribute = pd.json_normalize(df_business_restaurant.attributes)

# restaurants in Toronto with more than 100 reviews, rating above 3.5, accepting takeout, credit cards and price range is $$
df_toronto = df_business_restaurant.loc[df_business_restaurant['city']=='Toronto']
criteria = (df_toronto['stars'] > 3.5) & (df_toronto['review_count'] > 100) & \
           (df_toronto['is_open']==1) & (df_attribute.RestaurantsTakeOut=='True') & \
           (df_attribute.RestaurantsPriceRange2=='2') & (df_attribute.BusinessAcceptsCreditCards=='True')
# df_toronto_sub = df_toronto.loc[criteria]
# fig = plt.figure(figsize=(12, 6))
# sns.barplot(x='name', y='stars', data=df_toronto_sub.sort_values(by=['stars', 'review_count'], ascending=False)[0:15])

with open('Yelp Dataset/yelp_academic_dataset_tip.json') as file_tip:
    for i, line in enumerate(file_tip):
        tips.append(json.loads(line))
df_tips = pd.DataFrame(tips)
df_ri = df_business_restaurant.loc[(df_business_restaurant['name']=='Ramen Isshin') & criteria]
df_ri_tips = df_tips.loc[df_tips['business_id'].isin(df_ri.business_id)]


# replace in text
def text_prep(text):
    # filter out non-letters and transform them in lowercase
    text = re.sub('[^a-z\s]', '', text.lower())
    # filter stopwords
    text = [w for w in text.split() if w not in stopwords.words('english')]
    return ' '.join(text)


pd.set_option('mode.chained_assignment', None)
# apply function
df_ri_tips['text_cl'] = df_ri_tips['text'].apply(text_prep)
# create a word cloud
wc = WordCloud(width=1600, height=800, random_state=42, max_words=1000000)
wc.generate(str(df_ri_tips['text_cl']))
plt.figure(figsize=(15, 10), facecolor='black')
plt.title('Tips of Ramen Isshin', fontsize=40, color='white')
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=10)




