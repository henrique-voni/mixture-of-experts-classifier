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

#BaseEstimator, ClassifierMixin
class MixtureOfExperts():

    def __init__(self, estimators, gt, random_state):
        self._estimators = estimators
        self._gaussian_threshold = gt
        self._params = []
        self._pdfs = []
        self._X_per_cluster_idx = [] # Para armazenar o indice das amostras por cluster
        self._random_state = random_state
        self._km = KMeans(n_clusters = len(estimators), random_state=random_state)

    def _generate_params(self, X):
        self._km.fit(X)
        for i, center in enumerate(self._km.cluster_centers_):
            cluster_samples = X[self._km.labels_ == i]
            cluster_cov = np.cov(cluster_samples.T)
            self._params.append((center, cluster_cov))

    def _distribute_train(self, X, y):
        
        dist = []
        self._generate_params(X)
        for (center, cov) in self._params:
            
            pdf = self._mvpdf(X, center, cov)
            self._pdfs.append(pdf)

            rel_idx = np.argwhere(pdf > self._gaussian_threshold).flatten()
            X_rel = np.take(X, rel_idx, axis=0)
            y_rel = np.take(y, rel_idx, axis=0)
            # No treino os G's não fazem diferença porque não influenciam em qual amostra pegar, então basta retornar as amostras
            dist.append({"X" : X_rel, "y" : y_rel})
        return dist

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
        for estimator, (X_sub,y_sub) in zip(self._estimators, self._distribute_train(X,y)):
            estimator.fit(X_sub,y_sub)
        print("Estimators fitted successfully.")


    def predict(X):




