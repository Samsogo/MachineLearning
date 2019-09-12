'''
    为了检测模型的性能，我们不能使用全部数据进行训练模型，而是使用
    一部分数据作为测试数据，将原始数据分为训练数据和测试数据，训练
    数据占一大部分，然后使用测试数据进行测试模型性能。

    使用sklearn自带的数据集鸢尾花的数据集
'''


from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()

X = iris.data
Y = iris.target

#由于数据是按照标签顺序的，所以需要打乱训练
shuffle_index = np.random.permutation(len(X))

#测试集的比例
test_ratio = 0.2

size = int(len(X) * test_ratio)

test_index = shuffle_index[:size]
train_index = shuffle_index[size:]


#获取训练集
x_train = X[train_index]
y_train = Y[train_index]

#获取测试集
x_test = X[test_index]
y_test = Y[test_index]

knn = KNeighborsClassifier(n_neighbors=3)

#拟合训练
knn.fit(x_train,y_train)

#预测
y_predict = knn.predict(x_test)


#计算准确度
ratio = sum(y_predict==y_test)/len(y_test)

print(ratio)




