
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#加载数据集
digits = datasets.load_digits()

#获取特征
X = digits.data
#获取标签
Y = digits.target

#训练集和测试集分割
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=5)
#拟合进行训练
knn.fit(x_train,y_train)
#预测
y_predict = knn.predict(x_test)

#自己算的
acc = sum(y_predict==y_test)/len(y_test)

print(acc)

#sklearn自带的库
a = accuracy_score(y_test,y_predict)
print(a)

sc = knn.score(x_test,y_test)
print(sc)





