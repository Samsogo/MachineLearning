
'''
绘制散点图
'''

import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(0,1,10000)
y = np.random.normal(0,1,10000)
plt.scatter(x,y,alpha=0.1)
plt.show()

