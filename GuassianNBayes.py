from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples=800, centers=6, random_state=6)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
print('Training Score:', gnb.score(x_train, y_train))