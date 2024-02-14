import numpy as np

x = np.array([[0, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 1], [1, 1, 0, 1],
              [1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]])

y = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])

from sklearn.naive_bayes import BernoulliNB

# 参数
# alpha: 平滑参数
# binarize: 二值化界限，None值表示已经二值化
bnb = BernoulliNB()
bnb.fit(x, y)
day_pre = [[0, 0, 1, 0]]
print('预测结果：')
print('*' * 50)
print('结果：', bnb.predict(day_pre))
print("概率：", bnb.predict_proba(day_pre))
print('*' * 50)
