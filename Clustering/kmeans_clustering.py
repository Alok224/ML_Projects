import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

# To generate the sample data with 3 clusters
x, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)
# to check the shape of the samples
# print(x.shape)

# If I plot the sample data
plt.scatter(x[:,0],x[:,1])
# plt.show()  

# Splits the data 
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(x,y, test_size=0.33, random_state=42)
from sklearn.cluster import KMeans

# Manual process
# Elbow method to select the k value

wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k,init='k-means++')
    kmeans.fit(x_train)
    wcss.append(kmeans.inertia_)
# print(wcss)

# Now, draw the graph
# plt.plot(range(1,11),wcss)
# plt.xlabel('Number of clusters')
# plt.ylabel('wcss')
# plt.xticks(range(1,11))
# plt.show()

kmeans = KMeans(n_clusters = 3, init="k-means++")

# Fitting the model
y_labels = kmeans.fit_predict(x_train)
# print(y_labels)
plt.scatter(x_train[:,0],x_train[:,1],c=y_labels)
# plt.show()

y_test_labels = kmeans.predict(x_test)
# print(y_test_labels)

# to find the kvalue in automate way

from kneed import KneeLocator
kl = KneeLocator(range(1,11),wcss,curve = 'convex',direction='decreasing')
# print(kl.elbow)

# performance matrices
from sklearn.metrics import silhouette_score
silhouette_coefficients = []
for k in range(2,11):
    kmeans = KMeans(n_clusters=k,init="k-means++")
    kmeans.fit(x_train)
    score = silhouette_score(x_train,kmeans.labels_)
    silhouette_coefficients.append(score)

# print(silhouette_coefficients)

# draw the graph of silhoutte score 
plt.plot(range(2,11),silhouette_coefficients)
plt.xlabel('range(2,11)')
plt.ylabel('silhouette_coefficients')
plt.xticks(range(2,11))
# plt.show()