from sklearn.neighbors import KNeighborsClassifier
import numpy as np


knn = KNeighborsClassifier(n_neighbors=6)

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

#拟合过程
knn.fit(raw_data_X,raw_data_y)

#预测的值的列表
predict = knn.predict(x.reshape(1,-1))

#预测值
print(predict[0])