from sklearn.datasets import load_digits

digits = load_digits()
x = digits.data
y = digits.target
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=42,
                    learning_rate_init=.1)
clf.fit(x_train, y_train)
print("Training set score: %f" % clf.score(x_train, y_train))
print("Test set score: %f" % clf.score(x_test, y_test))
from sklearn.metrics import classification_report

print(classification_report(y_test, clf.predict(x_test)))
