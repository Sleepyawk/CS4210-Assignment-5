#-------------------------------------------------------------------------
# AUTHOR: Jonathan Lu
# FILENAME: clustering.py
# SPECIFICATION: K means and agglomerative clustering to find homogenity score for the dataset
# FOR: CS 4210 - Assignment #5
# TIME SPENT: 1hr 30 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('C:/Users/Administrator/Desktop/training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values)
silhouette_coefficient = 0
maxK = 0
# temp silhouette_coefficient
tempSC = 0

k_arr = []
sc_arr = []

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
for k in range(2,21,1):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
    silhouette_coefficient = silhouette_score(X_training, kmeans.labels_)
    if silhouette_coefficient > tempSC:
        maxK = k
        tempSC = silhouette_coefficient
    k_arr.append(k)
    sc_arr.append(silhouette_coefficient)

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
plt.plot(k_arr,sc_arr)
plt.show()

#reading the validation data (clusters) by using Pandas library
#--> add your Python code here
df = pd.read_csv('C:/Users/Administrator/Desktop/testing_data.csv', sep=',', header=None)
#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(df.values).reshape(1, len(df))[0]

#Calculate and print the Homogeneity of this kmeans clustering
# print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#rung agglomerative clustering now by using the best value o k calculated before by kmeans
#Do it:
agg = AgglomerativeClustering(n_clusters=maxK, linkage='ward')
agg.fit(X_training)

#Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())