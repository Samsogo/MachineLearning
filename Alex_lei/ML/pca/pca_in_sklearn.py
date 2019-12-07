from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from time import *
import matplotlib.pyplot as plt
import numpy as np


digits = datasets.load_digits()
x = digits.data
y = digits.target

x_train,x_test,y_train,y_test = train_test_split(x,y)
knn = KNeighborsClassifier(n_neighbors=5)

start = time()
knn.fit(x_train,y_train)
end = time()
score = knn.score(x_test,y_test)
print("降维之前的分类准确度：",score)
print("花费的时间：",(end-start)*1000)


pca = PCA(n_components=2)
pca.fit(x_train,y_train)
x_train_reduction = pca.transform(x_train)
x_test_reduction = pca.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=5)
start = time()
knn.fit(x_train_reduction,y_train)
end = time()
score_bak = knn.score(x_test_reduction,y_test)
print("降维之后的分类准确度：",score_bak)
print("花费的时间：",(end-start)*1000)

#会发现分类准确度没有那么高，但是时间少了不少，所以我们想找一个值尽可能使得分类准确度更高，使用的维度较少

pca = PCA(x_train.shape[1])
pca.fit(x_train,y_train)
print(pca.explained_variance_ratio_)  #该结果表示每个主成分可以解释的方差

a = [i for i in range(x_train.shape[1])]
b = [np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(len(pca.explained_variance_ratio_))]
plt.plot(a,b)
plt.show()  #从这个图大概可以看出当特征个数在25～30之间的时候就长的幅度很慢了，保留了原样本信息的增长幅度很小了

pca = PCA(0.96)  #该参数表示可以解释原数据95%以上的方差，保留95%以上原样本的信息

pca.fit(x_train,y_train)
final_x_train_reduction = pca.transform(x_train)
final_x_test_reduction = pca.transform(x_test)
print(len(pca.components_))  #该值等于31说明将数据的维度降到31个就可以解释原样本95%以上的方差

start = time()
knn.fit(final_x_train_reduction,y_train)
end = time()
final_score = knn.score(final_x_test_reduction,y_test)
print("降维之后的分类准确度：",final_score)
print("花费的时间：",(end-start)*1000)
