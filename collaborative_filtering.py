#-------------------------------------------------------------------------
# AUTHOR: Jonathan Lu
# FILENAME: collaborative_filtering.py
# SPECIFICATION: Collaborative filtering to read dataset to make user-based recommendations
# FOR: CS 4210 - Assignment #5
# TIME SPENT: 2 hr
#-----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('C:/Users/Administrator/Desktop/trip_advisor_data.csv', sep=',', header=0) #reading the data by using the Pandas library ()

#iterate over the other 99 users to calculate their similarity with the active user (user 100) according to their category ratings (user-item approach)
   # do this to calculate the similarity:
   #vec1 = np.array([[1,1,0,1,1]])
   #vec2 = np.array([[0,1,0,1,1]])
   #cosine_similarity(vec1, vec2)
   #do not forget to discard the first column (User ID) when calculating the similarities
   #--> add your Python code here
user_ids = df['User ID'].to_numpy()
df = df.drop(columns=['User ID'])
user_data = df.to_numpy()[:-1]
active = np.array([df.to_numpy()[-1]])
dict = {}

for i in range(len(active[0])):
    if active[0][i] == '?':
        active[0][i] = 0

for i in range(len(user_data)):
    dict[user_ids[i]] = np.array([user_data[i]])

similarities = {}

for i in range(len(user_data)):
    user = user_ids[i]
    vec1 = active
    vec2 = dict[user]
    cos_similarity = cosine_similarity(vec1, vec2)
    similarities[cos_similarity[0,0]] = user

#find the top 10 similar users to the active user according to the similarity calculated before
#--> add your Python code here
top_ten = sorted(similarities.keys())[-10:]
top_ten_users = []
for weight in top_ten:
    top_ten_users.append(similarities[weight])


   #Compute a prediction from a weighted combination of selected neighbors’ for both categories evaluated (galleries and restaurants)
   #--> add your Python code here
active_values = active[0]
similarity_sum = 0
gallery_index = 0
restaurant_index = 3

def get_sum(values, indexs):
    sum = 0
    for i in range(len(values)):
        if i in indexs:
            continue
        sum += values[i]

    return sum

# Gallery Prediction
indexs = [gallery_index, restaurant_index]
average_active_gallery = get_sum(active_values, indexs) / (len(active_values)-len(indexs))

for i in range(len(top_ten)):
    user = top_ten_users[i]
    weight = top_ten[i]

    values = [float(i) for i in dict[user][0]]
    gallery_value = values[gallery_index]

    user_sum = get_sum(values, indexs) / (len(active_values)-2)
    similarity_sum += (weight * (float(gallery_value) - user_sum))

weight_sum = sum(top_ten)
gallery_prediction = average_active_gallery + (similarity_sum / weight_sum)
active_values[gallery_index] = gallery_prediction

# Restaurant Predictiion
indexs = [restaurant_index]
average_active_restaurant = get_sum(active_values, indexs) / (len(active_values)-len(indexs))
similarity_sum = 0

for i in range(len(top_ten)):
    user = top_ten_users[i]
    weight = top_ten[i]
    values = [float(i) for i in dict[user][0]]
    restaurant_value = values[restaurant_index]

    user_sum = get_sum(values, indexs) / (len(active_values)-len(indexs))
    similarity_sum += (weight * (restaurant_value - user_sum))

restaurant_prediction = average_active_restaurant + (similarity_sum / weight_sum)

print('Gallery Prediction Weight: ', gallery_prediction)
print('Restaurant Prediction Weight: ', restaurant_prediction)