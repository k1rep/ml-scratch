from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=33)
# 认为存在线性关系
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)
# 模型性能评估，使用R2评估
from sklearn.metrics import r2_score
print(r2_score(y_test, lr_y_predict))

