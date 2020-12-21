import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

X,y = load_iris(return_X_y=True)

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=10)

kmeans.fit(X)

groups = []

for i in range(n_clusters):
    groups_idx = np.argwhere(kmeans.labels_ == i)
    groups.append(np.take(X, groups_idx, axis=0))


