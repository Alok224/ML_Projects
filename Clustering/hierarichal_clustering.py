import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

df = load_iris()
dataset = pd.DataFrame(data = df.data, columns=df.feature_names)
# print(dataset)
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
scaled_data = std.fit_transform(df.data)
# print(scaled_data)
# print(scaled_data.shape)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_scaled  = pca.fit_transform(scaled_data)
# plt.scatter(pca_scaled[:,0], pca_scaled[:,1])
# plt.show() 

# Now, apply the hierichal clustering(Aglomerative clustering)
# To draw the dendogram
import scipy.cluster.hierarchy as sc
# plot the dendogram
# plt.figure(figsize=(10,8))
# plt.title("dendogram")

# create dendogram
# sc.dendrogram(sc.linkage(pca_scaled,method = 'ward'))
# plt.title('Dendogram')
# plt.xlabel('Sample Index')
# plt.ylabel('Eculedian Distance')
# plt.show()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, linkage='ward', metric='euclidean')
cluster.fit(pca_scaled)

# print(cluster.labels_)
# plt.scatter(pca_scaled[:,0],pca_scaled[:,1],c = cluster.labels_)
# plt.show()


from sklearn.metrics import silhouette_score
silhouette_scores = []
for k in range(2,11):
    cluster = AgglomerativeClustering(n_clusters=k, linkage='ward', metric='euclidean')
    cluster.fit(pca_scaled)
    score = silhouette_score(pca_scaled, cluster.labels_)
    silhouette_scores.append(score)


plt.plot(range(2,11),silhouette_scores)
plt.xticks(range(2,11))
plt.xlabel("Number of clusters")
plt.ylabel("silhouette scores")
plt.show()