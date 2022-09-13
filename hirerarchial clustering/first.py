# import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs

from sklearn.manifold import MDS

DataSet, y = make_blobs(n_samples=10, centers=5, cluster_std=.8)

mds = MDS(n_components=2)
X_r = mds.fit(DataSet).embedding_
print(X_r)



linkage_data = linkage(DataSet, metric='euclidean')

# Scatter Plotting the the agglomerative cluster

fc = fcluster(linkage_data, 2, criterion='distance')
plt.figure(figsize=(5,5))
plt.scatter(X_r[:,0],X_r[:,1],c=fc)
plt.show()


# print("Sample scattering")
# print(linkage_data[:, 0], linkage_data[:, 1])
# plt.scatter(
#    linkage_data[:, 0], linkage_data[:, 1],
#    c='black', marker='o',
#    edgecolor='black', s=50
# )
# plt.title("Agglomerative Clustering Scattering")
# plt.xlabel("x axis")
# plt.ylabel("y axis")
# plt.show()

dendrogram(linkage_data)

# Plotting Dendogram of agglomerative clustering
plt.title("Agglomerative Clustering")
plt.xlabel("x axis")
plt.ylabel("y axis")

plt.show() 
