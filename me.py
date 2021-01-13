import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean


from sklearn.base import BaseEstimator, ClassifierMixin


GATING_THRESHOLD = .3
N_EXPERTS = 3

X,y = load_iris(return_X_y=True)

# def _rbf(x, center, cov):
#     dist = euclidean(x, center) ** 2
#     t = 2 * np.linalg.det(cov)
#     return np.exp( - dist / t )

# def rbf(X, center, cov):
#     return np.apply_along_axis(_rbf, 1, X, center, cov)

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

class MixtureModels(BaseEstimator, ClassifierMixin):

    def __init__(self, estimators, gating_bound, random_state):
        self.estimators_ = estimators
        self.gating_bound_ = gating_bound
        self.params_ = []
        self.random_state_ = random_state

    def generate_params_(self, X, n_clusters):
        pass
    
    def distribute_samples_(self, X, y, center, cov):
        pass

    def mvpdf_single_(self, x, center, cov):

        k = len(x)
        det_cov = np.linalg.det(cov)
        inv_cov = np.linalg.inv(cov)

        o = 1 / np.sqrt( (2 * np.pi) ** k  * det_cov)
        p = np.exp( -.5 * ( np.dot( (x - center).T, inv_cov ).dot( x-center ) ))
        return o * p

    def mvpdf_(self, X, center, cov):
        return np.apply_along_axis(self.mvpdf_single_, 1, X, center, cov)

    def softmax_(self, Z):
        exps = np.exp(Z - np.max(Z))
        return exps / exps.sum()

    def compute_g_(self, P):
        return self.softmax_(P) 

    def fit(self, X, y):
        self.params_ = self.generate_params_(X, len(self.estimators_))

        #params = (center, cov)
        for estimator, (center, cov) in zip(self.estimators_, self.params_):
            X_est, y_est = self.distribute_samples_(X,y, center, cov)
            estimator.fit(X_est, y_est)



            







# params = _generate_parameters(X, 3)

# # center1, cov2 = params[0]
# for center, cov in params:
#     pdf = rbf(X, center, cov)
#     # s = compute_g(pdf)
#     # rbf = rbf(X, center, cov)
#     # print(f"Max:{max(pdf)}")
#     # print(f"Min:{min(pdf)}")
#     print(pdf.shape)
#     print(max(pdf))
#     # print(f"Max: {max(rbf)}")


