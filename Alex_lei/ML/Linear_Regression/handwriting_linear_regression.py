import numpy as np
import matplotlib.pyplot as plot

X = np.array([1,2,3,4,5])
Y = np.array([1,3,2,5,4])

num = 0 #分子
fm  = 0 #分母
x_mean = np.mean(X)
y_mean = np.mean(Y)

for x,y in zip(X,Y):
    num += (x - x_mean)*(y - y_mean)
    fm += (x - x_mean) ** 2

a = num * 1.0 / fm
b = y_mean - a * x_mean

print(a)
print(b)

y = a * X + b

plot.scatter(X,Y,color='blue')
plot.plot(X,y,color='red')
plot.show()

