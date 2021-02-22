from me import MEClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from sklearn.neural_network import MLPClassifier
from sklearn_extensions.extreme_learning_machines import ELMClassifier

import numpy as np

import copy
import itertools

RUNS = 50

X,y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

n_experts = [5,10,15,20,25]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Cria os classificadores com base no modelo passado como parâmetro [estimators]
def create_estimators(estimators, number):
    i = itertools.cycle(estimators)
    clfs = []
    for _ in range(number)
        clfs.append(copy.deepcopy(next(i)))
    return clfs


mean_scores = np.zeros((len(n_experts), len(thresholds)))

for n in n_experts:
    for t in thresholds:
        experiment_scores = np.zeros(50)
        for i in range(1,51): #1,51 apenas para aproveitar o random_state
            estimators = create_estimators([MLPClassifier(), ELMClassifier()], n) #TODO: criar funçao para inicializar estimators
            me = MEClassifier(estimators, t, random_state=i)
            me.fit(X_train, y_train)
            y_pred = me.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            experiment_scores[i] = acc
        mean_scores[n,t] = np.mean(experiment_scores) # armazena a média dos 50 experimentos para esta configuração

best_acc = np.unravel_index(mean_scores.argmax(), mean_scores.shape)        
print(f"Melhor resultado: {mean_scores[best_acc]} \n Configuração: {n_experts[best_acc[0]]} especialistas e threshold em {thresholds[best_acc[1]]}")