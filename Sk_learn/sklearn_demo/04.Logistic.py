import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1.加载数据
iris = datasets.load_iris()
X = iris.data[:, :2]  # 使用前两个特征
Y = iris.target
# np.unique(Y)   # out: array([0, 1, 2])

# 2.拆分测试集、训练集。
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# 设置随机数种子，以便比较结果。

# 3.标准化特征值
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# 4. 训练逻辑回归模型
logreg = linear_model.LogisticRegression(C=1e5, max_iter=3000)  # 最大迭代次数
logreg.fit(X_train_std, Y_train)

# 5. 预测
prepro = logreg.predict_proba(X_test_std)
acc = logreg.score(X_test_std, Y_test)

print(acc)
