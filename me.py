import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean


from sklearn.base import BaseEstimator, ClassifierMixin

GATING_THRESHOLD = .3
N_EXPERTS = 3

X,y = load_iris(return_X_y=True)


#BaseEstimator, ClassifierMixin
class MixtureOfExperts():

    def __init__(self, estimators, gt, random_state):
        self._estimators = estimators
        self._gaussian_threshold = gt
        self._params = []
        self._random_state = random_state
        self._km = KMeans(n_clusters = len(estimators), random_state=random_state)


    """
        Método auxiliar para geração dos parâmetros centro e matrizes de covariância associados a cada especialista.

        Parâmetros: 
            X - Amostras
    """
    def _generate_params(self, X):
        self._km.fit(X)
        for i, center in enumerate(self._km.cluster_centers_):
            cluster_samples = X[self._km.labels_ == i]
            cluster_cov = np.cov(cluster_samples.T)
            self._params.append((center, cluster_cov))

    """
        Método auxiliar para calcular a probabilidade de uma amostra seguir uma distribuição normal multivariada com parâmetros
        de centro e matriz de covariância.

        Parâmetros:
            x - vetor de amostra
            center - vetor de centróide
            cov - matriz de covariância
    """
    def _mvpdf_single(self, x, center, cov):
        k = len(x)
        det_cov = np.linalg.det(2 * cov) # Cov multiplicada por 2 para evitar valores muito baixos nas probabilidades
        inv_cov = np.linalg.inv(cov)

        o = 1 / np.sqrt( (2 * np.pi) ** k  * det_cov)
        p = np.exp( -.5 * ( np.dot( (x - center).T, inv_cov ).dot( x-center ) ))
        return o * p

    """
        Método para calcular a PDF multivariada para N amostras.

        Parâmetros:
            X - amostras
            center - vetor de centróide
            cov - matriz de covariância
    """
    def _mvpdf(self, X, center, cov):
        return np.apply_along_axis(self._mvpdf_single, 1, X, center, cov)

    """
        Método para calcular a função softmax de Z valores.

        Parâmetros:
            Z - valores de saída a serem aplicados na softmax
    """
    def _softmax(self, Z):
        exps = np.exp(Z - np.max(Z))
        return exps / exps.sum()

    """
        Método para distribuição de amostras para cada especialista no treinamento
        
        Parâmetros:
            X - Amostras
            y - classes de cada amostra
        
        Primeiro, o método gera os centros e matrizes de covariância associados a cada especialista/cluster. Em seguida,
        calcula-se a probabilidade usando a função densidade de probabilidade da distribuição normal multivariada para
        cada amostra. A amostra é aceita pelo cluster se ela for maior que um limiar pré-definido (_gaussian_threshold).
    """
    def _distribute_train(self, X, y):      
        dist = []
        self._generate_params(X)
        for (center, cov) in self._params:
            
            pdf = self._mvpdf(X, center, cov)

            rel_idx = np.argwhere(pdf > self._gaussian_threshold).flatten()
            X_rel = np.take(X, rel_idx, axis=0)
            y_rel = np.take(y, rel_idx, axis=0)
            # No treino os G's não fazem diferença porque não influenciam em qual amostra pegar, então basta retornar as amostras
            dist.append({"X" : X_rel, "y" : y_rel})
        return dist

    """
        Método para calcular a matriz de G's (amostras x especialistas)
    """
    def _compute_g(self, X):

        pdfs = [] #matriz de probabilidades P
        for (center, cov) in self._params:
            pdfs.append(self._mvpdf(X, center, cov))

        P = np.array(pdfs).T #matriz de probabilidades
        G = np.apply_along_axis(self._softmax, 0, P)
        return G
        


    def fit(self, X, y):   
        for estimator, (X_sub,y_sub) in zip(self._estimators, self._distribute_train(X,y)):
            estimator.fit(X_sub,y_sub)
        print("Estimators fitted successfully.")


    def predict(self, X):
        pass



