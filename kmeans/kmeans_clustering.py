from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1,2], [1,4], [1,0] , [10,2], [10,2], [10,0]])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

cluster_centers = kmeans.cluster_centers_

print("The Clusters are:")
for i in cluster_centers:
    print(i)
