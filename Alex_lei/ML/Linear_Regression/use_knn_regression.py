from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

boston = datasets.load_boston()
X = boston.data
Y = boston.target

X = X[Y<50]
Y = Y[Y<50]

x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=777)

knn_reg = KNeighborsRegressor()
knn_reg.fit(x_train,y_train)
score = knn_reg.score(x_test,y_test)
print(score)