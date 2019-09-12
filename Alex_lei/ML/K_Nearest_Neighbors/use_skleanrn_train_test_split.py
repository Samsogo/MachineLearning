from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data
Y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)

y_predict = knn.predict(x_test)

ratio = sum(y_predict == y_test)/len(y_test)

print(ratio)