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

mlp_single = MLPClassifier(hidden_layer_sizes=5, activation="relu", random_state=100)

mlp_me_1 = MLPClassifier(hidden_layer_sizes=5, activation="relu", random_state=100)
me_1 = MEClassifier(estimators=[mlp_me_1], gt=0.33, random_state=100)

print(f"MLP: {classify_dataset(X, y, mlp_single)}" )
print(f"ME Single MLP: {classify_dataset(X, y, me_1)}")


