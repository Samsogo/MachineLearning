'''

    多元线性回归算法
    数据集：波士顿房产数据集

'''

import numpy as np
from sklearn import datasets
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()
X = boston.data
Y = boston.target

X = X[Y<50]
Y = Y[Y<50]

x_train, x_test, y_train, y_test = train_test_split(X,Y)

x_b = np.hstack([np.ones((len(x_train),1)),x_train])

t = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)

print(t[0])
print(t[1:])

y_predict = np.hstack([np.ones((len(x_test),1)),x_test]).dot(t)
r = r2_score(y_test,y_predict)
print(r)

