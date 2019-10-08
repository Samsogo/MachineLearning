import numpy as np
import time

m = 1000000
x = np.random.random(size=m)
y = x * 2 + np.random.normal(size=m)

x_mean = np.mean(x)
y_mean = np.mean(y)

#向量化运算

time_start = time.time()

num = (x - x_mean).dot(y - y_mean)
fm = (x - x_mean).dot(x - x_mean)

time_end = time.time()

print("向量化运算时间:",time_end - time_start)


num = 0
fm = 0

time_start = time.time()

for i,j in zip(x,y):
    num += (i - x_mean) * (j - y_mean)
    fm += (i - x_mean) * (i - x_mean)

time_end = time.time()

print("普通运算时间:",time_end - time_start)


a = num * 1.0 / fm
b = y_mean - a * x_mean

print(a)
print(b)
