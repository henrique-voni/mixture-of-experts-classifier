import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

GATING_THRESHOLD = .3
N_EXPERTS = 3

X,y = load_iris(return_X_y=True)


def _multivariate_normal_pdf(x, center, cov):

    k = len(x)

    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)

    o = 1 / np.sqrt( (2 * np.pi) ** k  * det_cov)
    p = np.exp( -.5 * ( np.dot( (x - center).T, inv_cov ).dot( x-center ) ))
    return o * p

def multivariate_normal_pdf(X, center, cov):
    return np.apply_along_axis(_multivariate_normal_pdf, 1, X, center, cov)

def _softmax(Z):
    exps = np.exp(Z - np.max(Z))
    return exps / exps.sum()

def compute_g(P):
    return _softmax(P) 

def distribute_samples(X):
    
    centers = _generate_prototypes(X, N_EXPERTS)
    cov_x = np.cov(X.T)
    
    for prototype in centers:
        p_x = multivariate_normal_pdf(X, prototype, cov_x)
        g_x = compute_g(p_x)
        ## selecionar amostras baseado no threshold.


def _generate_prototypes(X, n_experts):
    kmeans = KMeans(n_clusters=n_experts, random_state=10)
    kmeans.fit(X)
    return kmeans.cluster_centers_

#     group_idx = np.argwhere(kmeans.labels_ == i)
#     group_samples = np.take(X, group_idx, axis=0)
