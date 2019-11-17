'''

    使用波士顿房产数据，使用梯度下降法计算我们的线性回归的回归参数

'''

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

#加载波士顿数据集
boston = datasets.load_boston()

#获取其特征和标签
X = boston.data
Y = boston.target

#去除错误数据
X = X[Y < 50]
Y = Y[Y < 50]

#将数据集分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=666)

#数据归一化，将数据处理成均值为0方差为1的数据，防止数据之间差值太大造成结果不准确
standard = StandardScaler()
standard.fit(x_train)
x_train_standard = standard.transform(x_train)
x_test_standard = standard.transform(x_test)

#使用梯度下降法进行计算回归参数，使用的是随机梯度下降法

#定义求偏导函数

def DJ(X_b_i, theta, y_i):
    return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2. / len(X_b_i)

#定义计算回归参数的主程序(随机梯度函数)

def random_gradient_descent(X_b, theta, y, n_iters, to, t1):
    def learning_rate(iters):
        return to / (iters + t1)

    m = len(X_b)
    for i in range(n_iters):
        indexes = np.random.permutation(m) #随机0～m的数字，打乱顺序的
        X_b_new = X_b[indexes]
        Y_new = y[indexes]
        for j in range(m):
            eta = learning_rate(i * m + j)
            theta = theta - eta * DJ(X_b_new[j],theta,Y_new[j])
    return theta

X_train_b = np.hstack([np.ones((len(x_train_standard),1)),x_train_standard]) #构建X_b，X_b就是在原有的特征矩阵中最前面在加全是1的一列
theta = np.zeros(X_train_b.shape[1]) #初始化一个theta矩阵
res_theta = random_gradient_descent(X_train_b, theta, y_train, 100, 5, 50)

#使用计算好的回归参数去预测测试集的标签值
X_test_b = np.hstack([np.ones((len(x_test_standard),1)),x_test_standard])
y_predict = X_test_b.dot(res_theta)

#使用r^2检验结果

score = r2_score(y_test, y_predict)
print("随机梯度下降法：",score)



#使用传统的方法计算回归参数
res_theta_bak = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
y_predict_bak = X_test_b.dot(res_theta_bak)
score_bak = r2_score(y_test, y_predict_bak)
print("传统方法：",score_bak)
