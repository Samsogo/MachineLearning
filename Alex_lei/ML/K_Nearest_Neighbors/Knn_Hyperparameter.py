'''
    所谓超参数，是在学习之前设置的参数，而不是通过训练得到的参数
    这里说三个参数
    k：KNN的k
    p：明可夫斯基距离公式的p
    weight：权重
'''

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
X = iris.data
Y = iris.target

shuffleindex = np.random.permutation(len(X))

ratio = 0.2
size = int(len(X) * ratio)

train_size = shuffleindex[size:]
test_size = shuffleindex[:size]

x_train = X[train_size]
y_train = Y[train_size]

x_test = X[test_size]
y_test = Y[test_size]


#k的确定
best_score = 0
best_k = 0
for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    score = knn.score(x_test,y_test)
    if score > best_score:
        best_score = score
        best_k = k

print(best_score)
print(best_k)

#weight

#weight的确定
best_weight = ''
for method in ['uniform','distance']:
    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k,weights=method)
        knn.fit(x_train, y_train)
        score = knn.score(x_test, y_test)
        if score > best_score:
            best_score = score
            best_k = k
            best_weight = method

print(best_score)
print(best_k)
print(best_weight)

#p的确定,当使用p的时候，weight必须是distance，但是默认是uniform

best_p = 0

for i in range(1,6):
    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k,weights='distance',p=i)
        knn.fit(x_train, y_train)
        score = knn.score(x_test, y_test)
        if score > best_score:
            best_score = score
            best_k = k
            best_p = i


print(best_score)
print(best_k)
print(best_p)