import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

m = 100
x = np.random.uniform(-5,5,size=m)
y = 2.5 * x ** 2 + 3 * x + 2 + np.random.normal(0,1,size=m)
X = x.reshape(-1,1)


#sklearn中封装了Pipeline这个类，可以将我们的流程串起来
del_pipe = Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('stand',StandardScaler()),
    ('lin_reg',LinearRegression())
])

del_pipe.fit(X,y)
y_predict = del_pipe.predict(X)
score = del_pipe.score(X,y)
print(score)
plt.scatter(x,y)
plt.plot(np.sort(x),y_predict[np.argsort(x)],c='r')
plt.show()

import sys
print(sys.version)



