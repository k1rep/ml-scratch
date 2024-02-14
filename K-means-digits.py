import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
x = digits.data
y = digits.target
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.cluster import KMeans
from sklearn import metrics


# 设置聚类函数，参数为分类数量
def julei(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    # 训练模型
    kmeans.fit(x_train)
    # 获取聚类中心
    y_pre = kmeans.predict(x_test)
    # 当数据本身带有正确的类别信息时，使用ARI指标性能评估
    print('聚类数为%d时，ARI指标为：%f' % (n_clusters, metrics.adjusted_rand_score(y_test, y_pre)))


julei(5)
julei(8)
julei(10)
julei(15)
julei(20)

from sklearn.metrics import silhouette_score
# 当数据不含标记信息时，使用轮廓系数度量
def lunkuo(n_clusters):
    lunkuo = list(range(2, n_clusters+1))
    sc_scores = []
    for i in lunkuo:
        cls = KMeans(n_clusters=i)
        cls.fit(x_train)
        sc_scores.append(silhouette_score(x_train, cls.labels_, metric='euclidean'))
    plt.figure()
    plt.plot(lunkuo, sc_scores)
    plt.xlabel('cluster numbers')
    plt.ylabel('silhouette score')
    plt.show()


lunkuo(20)