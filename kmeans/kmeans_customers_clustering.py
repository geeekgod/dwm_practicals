import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Data
df =  pd.read_csv("./mall_customers_data.csv")
print(df.shape)

df_2d = df[['Age', 'Annual Income (k$)']]
df_2d = df_2d.dropna(axis=0)

df.head()


df.drop(["CustomerID"], axis = 1, inplace=True)

wcss  = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_2d)
    wcss .append(kmeans.inertia_)

plt.plot(range(1,11), wcss )
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("wcss")
plt.show()

# 3D plot of KMeans Clustering
km = KMeans(n_clusters=6)
clusters = km.fit_predict(df.iloc[:,1:])
df["label"] = clusters

 
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c='blue', s=60)
ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c='red', s=60)
ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c='green', s=60)
ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c='orange', s=60)
ax.scatter(df.Age[df.label == 4], df["Annual Income (k$)"][df.label == 4], df["Spending Score (1-100)"][df.label == 4], c='purple', s=60)
ax.scatter(df.Age[df.label == 5], df["Annual Income (k$)"][df.label == 5], df["Spending Score (1-100)"][df.label == 5], c='black', s=60)
ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()


km = KMeans(n_clusters=6)
clusters = km.fit_predict(df.iloc[:,1:])
df["label"] = clusters

 
fig = plt.figure(figsize=(20,10))
plt.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0],  c='blue', s=60)
plt.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], c='red', s=60)
plt.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], c='green', s=60)
plt.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], c='orange', s=60)
plt.scatter(df.Age[df.label == 4], df["Annual Income (k$)"][df.label == 4], c='purple', s=60)
plt.scatter(df.Age[df.label == 5], df["Annual Income (k$)"][df.label == 5], c='black', s=60)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
plt.show()