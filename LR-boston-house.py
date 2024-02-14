from sklearn.datasets import load_boston
bostonHouse = load_boston()
x = bostonHouse.data
y = bostonHouse.target
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=5)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xTrain, yTrain)
lr_y_predict = lr.predict(xTest)
from sklearn.metrics import mean_squared_error
print("MSE: ", mean_squared_error(yTest, lr_y_predict))