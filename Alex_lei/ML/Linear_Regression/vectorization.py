import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5,6])
y = np.array([1,3,2,5,4,6])

x_mean = np.mean(x)
y_mean = np.mean(y)

#向量化运算


num = (x - x_mean).dot(y - y_mean)
fm = (x - x_mean).dot(x - x_mean)


a = num * 1.0 / fm
b = y_mean - a * x_mean

Y = a * x + b


plt.scatter(x,y,color = "red")
plt.plot(x,Y,color='blue')
plt.show()

print(a)
print(b)
