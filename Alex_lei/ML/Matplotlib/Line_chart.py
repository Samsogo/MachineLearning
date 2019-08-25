'''
绘制折线图
'''

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,10000)
y = (x/5).copy()

sinx = np.sin(x)
cosx = np.cos(x)

plt.plot(x,sinx,color='red',linestyle='--',label='sin(x)')
plt.plot(x,cosx,color='green',label='cos(x)')
plt.plot(x,y,color='blue',label='y=ax+b')

plt.xlabel("sin")
plt.ylabel("cos")
plt.legend()
plt.title("Welcome to MachineLearning")
plt.show()
