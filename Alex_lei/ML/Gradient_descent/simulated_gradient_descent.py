'''

    模拟实现梯度下降法

'''

import numpy as np
import matplotlib.pyplot as plt


plot_x = np.linspace(-1,6,141) #等间距初始化141个点
plot_y = (plot_x - 2.5) ** 2 -1

#求导数
def DJ(theta):
    return 2.5 * (theta - 2.5)

#损失函数J
def J(theta):
    return (theta - 2.5) ** 2 - 1

#模拟梯度下降法
def Simulated_gradient_descent(theta,eta, n_iters = 1000):
    theta_history = []

    while True:
        last_theta = theta
        theta = theta - eta * DJ(theta)
        theta_history.append(theta)
        if np.abs(J(theta) - J(last_theta)) < 1e-8:
            break
    return theta,theta_history

#主函数
if __name__ == '__main__':
    theta = 0
    '''
        eta称为为学习率
        eta的取值影响我们获取最优解的效率
        eta取值不当可能使的我们获取不到最优解
        eta是梯度下降法中的一个超参数
    '''
    eta = 0.01
    theta_res,theta_his = Simulated_gradient_descent(theta,eta)
    print(theta_res)
    theta_x = np.array(theta_his)
    theta_y = J(theta_x)
    plt.plot(plot_x,plot_y)
    plt.plot(theta_x,theta_y,color = 'r',marker = '+')
    plt.show()
