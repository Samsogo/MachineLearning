import numpy as np
from math import sqrt
from collections import Counter
import matplotlib.pyplot as plt


#训练集的特征
raw_data_X = [[1.232422,1.22324],
              [2.324232,1.3224],
              [2.3435353,2.3232342],
              [3.434353,3.434353],
              [4.54546,3.54544],
              [7.42422,6.764353],
              [6.42224534,7.533232],
              [8.435353,8.5433],
              [9.423534,9.422224],
              [8.544444,9.4564454]]
#训练集的标签
raw_data_y=[0,0,0,0,0,1,1,1,1,1]

#测试集
x = np.array([7.5353343,8.53324232])

x_train = np.array(raw_data_X)
y_train = np.array(raw_data_y)

plt.scatter(x_train[y_train==0,0],x_train[y_train==0,1],color='r')
plt.scatter(x_train[y_train==0,0],x_train[y_train==1,1],color='g')
plt.scatter(x[0],x[1],color='b')
plt.show()

distance = []

#计算每个点到测试集的距离
for xt in x_train:
    d = sqrt(np.sum((xt-x)**2))
    distance.append(d)

#取前6个较小的
k = 6
pre_arr = [y_train[i] for i in np.argsort(distance)[:k]]


cal_arr = Counter(pre_arr)
print(cal_arr)

#预测值
predict = cal_arr.most_common(1)
print(predict[0][0])