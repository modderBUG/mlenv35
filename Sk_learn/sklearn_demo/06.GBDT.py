import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1.加载数据
iris = datasets.load_iris()
X = iris.data[:, :4]  # 使用前两个特征
Y = iris.target
# 分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score

from sklearn import metrics

model = GradientBoostingClassifier(n_estimators=10)
model.fit(X_train,
          y_train)


#预测
y_test, y_pred = y_test, model.predict(X_test)
print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
# y_train_proba = model.predict_proba(X_train)[:,1]
# print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_train_proba))
# y_proba = model.predict_proba(X_test)[:,1]
# print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_proba))
#
#
# #预测
# y_test, y_pred = y_test, model.predict(X_test)
# print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_pred))
# y_train_proba = model.predict_proba(X_train)[:,1]
# print("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_train_proba))
# y_proba = model.predict_proba(X_test)[:,1]
# print("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_proba))