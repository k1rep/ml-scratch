from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=30000, centers=6, random_state=6)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=33, test_size=0.25)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)
x_test = scaler.transform(x_test)
model = MultinomialNB()
model.fit(x_train_s, y_train)
print("准确率：", model.score(x_test_s, y_test))
