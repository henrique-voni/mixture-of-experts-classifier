import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean


from sklearn.base import BaseEstimator, ClassifierMixin

GATING_THRESHOLD = .3
N_EXPERTS = 3

X,y = load_iris(return_X_y=True)

# def _mvpdf(x, center, cov):
#     inv_cov = np.linalg.inv(cov)
#     return np.exp( -.5 * ( np.dot( (x - center).T, inv_cov ).dot( x-center ) ))

# def mvpdf(X, center, cov):
#     return np.apply_along_axis(_mvpdf, 1, X, center, cov)

# def distribute_samples(X):
    
#     params = _generate_parameters(X, N_EXPERTS)
    
#     for center, cov_matrix in params:
#         p_x = multivariate_normal_pdf(X, center, cov_matrix)
#         p_x_rel = np.take(p_x, np.argwhere(p_x > p_x.mean()))
#         g_x = compute_g(p_x_rel)
#         print(f"Max: {max(g_x)}")
#         # print(f"Sum: {sum(g_x)}")
        
#         ## selecionar amostras baseado no threshold.



# def _generate_parameters(X, n_experts):
#     kmeans = KMeans(n_clusters=n_experts, random_state=10)
#     kmeans.fit(X)
#     params = []

#     for i, center in enumerate(kmeans.cluster_centers_):
#         group_idx = np.argwhere(kmeans.labels_ == i).flatten()
#         group_samples = np.take(X, group_idx, axis=0)
#         group_cov = np.cov(group_samples.T)
#         params.append((center, group_cov))    
#     return params

class MixtureOfExperts(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators, gt, random_state):
        self._estimators = estimators
        self._gaussian_threshold = gt
        self._params = []
        self._random_state = random_state
        self._km = KMeans(n_clusters = len(estimators), random_state=random_state)

    def _generate_params(self, X):
        self._km.fit(X)
        for i, center in enumerate(self._km.cluster_centers_):
            cluster_samples = X[self._km.labels_ == i]
            cluster_cov = np.cov(cluster_samples.T)
            self._params.append((center, cluster_cov))

    
    def _distribute_samples(self, X, y):
        pass

    def _mvpdf_single(self, x, center, cov):
        k = len(x)
        det_cov = np.linalg.det(2 * cov) # Cov multiplicada por 2 para evitar valores muito baixos nas probabilidades
        inv_cov = np.linalg.inv(cov)

        o = 1 / np.sqrt( (2 * np.pi) ** k  * det_cov)
        p = np.exp( -.5 * ( np.dot( (x - center).T, inv_cov ).dot( x-center ) ))
        return o * p

    def _mvpdf(self, X, center, cov):
        return np.apply_along_axis(self._mvpdf_single, 1, X, center, cov)

    def _softmax(self, Z):
        exps = np.exp(Z - np.max(Z))
        return exps / exps.sum()

    def _compute_g(self, P):
        return self._softmax(P) 

    def fit(self, X, y):   
        pass

    def predict(X):
        pass


