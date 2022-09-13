# import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

DataSet, y = make_blobs(
   n_samples=10, n_features=2,
   centers=3, cluster_std=0.5,
   shuffle=True, random_state=0
)


linkage_data = linkage(DataSet, metric='euclidean')

# Scatter Plotting the the agglomerative cluster
plt.scatter(
   linkage_data[:, 0], linkage_data[:, 1],
   c='white', marker='o',
   edgecolor='black', s=50
)
plt.title("Agglomerative Clustering Scattering")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()

dendrogram(linkage_data)

# Plotting Dendogram of agglomerative clustering
plt.title("Agglomerative Clustering")
plt.xlabel("x axis")
plt.ylabel("y axis")

plt.show() 
