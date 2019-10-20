'''

     采用波士顿房产数据进行多元线性回归分析

'''

from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()
X = boston.data
Y = boston.target

X = X[Y<50]
Y = Y[Y<50]

x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=666)

lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)
y_predict = lin_reg.predict(x_test)
score = lin_reg.score(x_test,y_test)
print(score)
print(lin_reg.coef_)
print(lin_reg.intercept_)