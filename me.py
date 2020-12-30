import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvnp


X,y = load_iris(return_X_y=True)

n_clusters = 3
# kmeans = KMeans(n_clusters=n_clusters, random_state=10)

# kmeans.fit(X)

# groups = []

# for (i, center) in enumerate(kmeans.cluster_centers_):
#     group_idx = np.argwhere(kmeans.labels_ == i)
#     group_samples = np.take(X, group_idx, axis=0)
#     groups.append( (center, group_samples) )


def multivariate_normal(X, center, cov):

    # X = input vector (1, n_features)
    # center = mean parameter
    # cov = variance-covariance matrix parameter

    k = len(X)

    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)

    o = 1 / np.sqrt( (2 * np.pi) ** k  * det_cov)
    p = np.exp( -.5 * ( np.dot( (X - center).T, inv_cov ).dot( X-center ) ))
    return o * p

def softmax(Z):
    exps = np.exp(Z - np.max(Z))
    return exps / np.sum(exps)

mean = np.mean(X, axis=0)
cov = np.cov(X.T)


p = mvnp.pdf(X, cov=np.cov(X.T), mean=mean)

b = multivariate_normal(X[0],mean, cov)
print(b)
print(p[0])