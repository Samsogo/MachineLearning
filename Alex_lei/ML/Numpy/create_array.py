

'''
numpy的数组的创建
'''


import numpy as np


l1 = np.arange(5) #创建有规律的一维数组（元组构成）
print(type(l1))
print(l1)

l2 = np.array((1,23,456,65,43)) #创建一个没有规律的一维数组（元组构成）
print(type(l2))
print(l2)

l3 = np.array([1,2,3,45,67]) #创建一个没有规律的一维数组（列表构成）
print(type(l3))
print(l3)

l4 = np.array(((1,2,3),(4,5,6),(7,8,9))) #创建一个没有规律的二维数组（元组构成）
print(type(l4))
print(l4)

l5 = np.array([[1,2,3],[4,5,6],[7,8,9]]) #创建一个没有规律的二维数组（列表构成）
print(type(l5))
print(l5)

l6 = np.zeros(3) #返回一个全是0的，大小为3的一维数组
print(type(l6))
print(l6)

l7 = np.zeros([3,4]) #返回一个全是0的，大小为3*4的二维数组
print(type(l7))
print(l7)
print(l7.size)