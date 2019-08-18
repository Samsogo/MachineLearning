import numpy as np

arr = np.arange(10) #初始化一个数组
x = arr
np.random.shuffle(arr) #将数组进行打乱顺序
print(arr) #打印数组
print(np.sort(arr))  #对数组进行从小到大排序
print(np.argsort(arr)) #打印的是从小到大的元素所在的下标

print(np.partition(x,4)) #分区函数，类似于快速排序，4就是我们所选的基数，4之前的都比4小，4之后的都比4大。