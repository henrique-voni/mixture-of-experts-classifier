from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from me import MEClassifier
from sklearn.neural_network import MLPClassifier
from sklearn_extensions.extreme_learning_machines import ELMClassifier

def classify_dataset(X, y, estimator):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=100)
    estimator.fit(X_train,y_train)
    y_pred = estimator.predict(X_test)
    return accuracy_score(y_test, y_pred)

X, y = load_iris(return_X_y=True)


## Caso 1: 1 MLP x ME com 1 MLP (teoricamente, o desempenho deve ser o mesmo em ambos os casos).
mlp_single = MLPClassifier(hidden_layer_sizes=5, activation="relu", random_state=100, max_iter=10000)

mlp_me_1 = MLPClassifier(hidden_layer_sizes=5, activation="relu", random_state=100, max_iter=10000)
me_1 = MEClassifier(estimators=[mlp_me_1], gt=0.33, random_state=100)

print(f"MLP: {classify_dataset(X, y, mlp_single)}" )
print(f"ME Single MLP: {classify_dataset(X, y, me_1)}")


## Caso 2: 1 MLP x ME com 3 MLPs
mlp_single_2 = MLPClassifier(hidden_layer_sizes=3, activation="relu", random_state=100, max_iter=10000)

mlp_me_2 = MLPClassifier(hidden_layer_sizes=3, activation="relu", random_state=100, max_iter=10000)
mlp_me_3 = MLPClassifier(hidden_layer_sizes=3, activation="relu", random_state=100, max_iter=10000)
mlp_me_4 = MLPClassifier(hidden_layer_sizes=3, activation="relu", random_state=100, max_iter=10000)
me_2 = MEClassifier([mlp_me_2, mlp_me_3, mlp_me_4], gt=0.33, random_state=100)

print(f"MLP: {classify_dataset(X, y, mlp_single_2)}" )
print(f"ME w/ 3 MLPs: {classify_dataset(X, y, me_2)}")

## Caso 3: 1 ELM x ME com 3 ELMs
elm_single_1 = ELMClassifier(n_hidden=4, random_state=100)

elm_me_1 = ELMClassifier(n_hidden=4, random_state=100)
elm_me_2 = ELMClassifier(n_hidden=4, random_state=100)
elm_me_3 = ELMClassifier(n_hidden=4, random_state=100)
me_3 = MEClassifier([elm_me_1, elm_me_2, elm_me_3], gt=0.33, random_state=100)

print(f"ELM: {classify_dataset(X, y, elm_single_1)}")
print(f"ME w/ 3 ELM: {classify_dataset(X, y, me_3)}")