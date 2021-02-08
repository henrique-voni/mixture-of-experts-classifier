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


    def _normalize(self, Z):
        return Z / Z.sum()


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

        all_rel_idx = [] # Lista de amostras selecionadas
        unselected_idx = []
        pdfs = [] # armazena as probabilidades para montar a matriz que selecionará as amostras não usadas

        self._generate_params(X)

        for (center, cov) in self._params:
            
            pdf = self._mvpdf(X, center, cov)
            pdfs.append(pdf)
            rel_idx = np.argwhere(pdf > self._gaussian_threshold).flatten()

            all_rel_idx.append(rel_idx)

            X_rel = np.take(X, rel_idx, axis=0)
            y_rel = np.take(y, rel_idx, axis=0)

            dist.append({"X" : X_rel, "y" : y_rel})


        P = np.vstack(pdfs).T #concatena todas as PDFS em uma única matriz

        all_rel_idx = np.unique(np.concatenate([arr for arr in all_rel_idx])) # cria um vetor com as amostras usadas
        unselected_idx = np.setdiff1d(np.arange(len(y)), all_rel_idx) # cria um vetor com as amostras não usadas por nenhum cluster

        unselected_P = np.take(P, unselected_idx, axis=0) #valores das PDF apenas para as amostras não selecionadas
        unselected_dist_idx = np.argmax(unselected_P, axis=1) #verifica qual PDF tem maior valor pelo indice
        
        for i, _ in enumerate(dist):

            uns_X = X[np.argwhere(unselected_dist_idx == i).flatten()]
            uns_y = y[np.argwhere(unselected_dist_idx == i).flatten()]
            dist[i]["X"] = np.concatenate((dist[i]["X"], uns_X))
            dist[i]["y"] = np.concatenate((dist[i]["y"], uns_y))

        return dist


    """
        Método para calcular a matriz de G's (amostras x especialistas)

        Parâmetros:
            - X: amostras
            - mode ["softmax", "normal"]: modo como os G's serão gerados para gerar valores entre 0 e 1. "softmax" aplica a função
            softmax sobre a entrada, enquanto "normal" faz uma simples divisão do valor sobre o total dos valores.
    """
    def _compute_g(self, X, norm_mode="softmax"):
        
        if not self._params:
            self._generate_params(X)
        pdfs = [] #matriz de probabilidades P
        
        for (center, cov) in self._params:
            pdfs.append(self._mvpdf(X, center, cov))
        P = np.array(pdfs).T #matriz de probabilidades

        if norm_mode == "softmax":
            return np.apply_along_axis(self._softmax, 0, P)
        return np.apply_along_axis(self._normalize, 0, P)
        


    def fit(self, X, y):   
        self._classes = np.unique(y)
        for estimator, dist in zip(self._estimators, self._distribute_train(X,y)):
            estimator.fit(dist["X"],dist["y"])
        print("Estimators fitted successfully.")


    def predict(self, X, mode="softmax"):        
        
        G = self._compute_g(X, mode)
        pred = np.zeros( (len(self._estimators), X.shape[0]), dtype=int ) 
        
        for i, estimator in enumerate(self._estimators):
            y_est = estimator.predict(X)        
            pred[i] = y_est

        pred = pred.T # Matriz de predições (amostras x especialistas)
        C = len(self._classes) # Quantidade de classes
        M, N = pred.shape

        row, col = np.indices((M,N))
        P3d = np.zeros(shape=(M,N,C))
        P3d[row, col, pred-1] = G
        P = P3d.sum(axis=1)
        return np.argmax(P, axis=1)


X, y = load_iris(return_X_y=True)
from sklearn.neural_network import MLPClassifier

mlp_1 = MLPClassifier(hidden_layer_sizes=2)
mlp_2 = MLPClassifier(hidden_layer_sizes=2)

me = MixtureOfExperts(estimators=[mlp_1, mlp_2], gt=0.003, random_state=10)

me.fit(X,y)
A = me.predict(X,y)

from sklearn.metrics import accuracy_score

print(accuracy_score(A, y))