import numpy as np

arr = np.array(np.arange(12).reshape(3,4))
print(arr)
print(arr[0]) #获取二维数组的第一行

print(arr[1]) #获取二维数组的第二行

print(arr[:3]) #获取二维数组的前三行

print(arr[[0,2]]) #获取二维数组的第1行和第三行

print(arr[:,1]) #获取二维数组的第二列

print(arr[:,-2:]) #获取二维数组的后两列

print(arr[:,[0,2]]) #获取二维数组的第一列和第三列

print(arr[1,3]) #获取二维数组的第二行第四列的元素