from sklearn.datasets import fetch_20newsgroups
news_groups = fetch_20newsgroups(subset='all')
x = news_groups.data
y = news_groups.target
# 查看目标变量名称
print(news_groups.target_names)
# 查看特征变量情况
print(news_groups.DESCR)
# 查看目标变量情况
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=33)

# 文本向量化
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
x_vec_train = vec.fit_transform(x_train)
x_vec_test = vec.transform(x_test)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_vec_train, y_train)
y_predict = mnb.predict(x_vec_test)

from sklearn.metrics import classification_report
print("Accuracy:", mnb.score(x_vec_test, y_test))
print(classification_report(y_test, y_predict, target_names=news_groups.target_names))
