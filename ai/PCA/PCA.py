# 数据载入
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
data = pd.read_csv('d:/ai/PCA/iris.csv')
print('查看数据前几行：\n',data.head())

X = data.drop(['class'],axis=1)
y = data.loc[:,'class']
print('统计标签个数：\n',pd.Series(y).value_counts())

# 用KNN算法进行分类
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X,y)
y_predict = KNN.predict(X)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y,y_predict)
print('KNN准确率：\n',accuracy)

# 数据标准化处理
from sklearn.preprocessing import StandardScaler
X_norm = StandardScaler().fit_transform(X)
print('标准化数据：\n',X_norm)

# 计算均值与方差
x1_mean = X.loc[:,'sepallength'].mean()
x1_norm_mean = X_norm[:,0].mean()
x1_sigma = X.loc[:,'sepallength'].std()
x1_norm_sigma = X_norm[:,0].std()
print('前后均值与方差：\n',x1_mean,x1_sigma,x1_norm_mean,x1_norm_sigma)

# 可视化
plt.figure(figsize=(18,12))
plt.subplot(231)
plt.hist(X.loc[:,'sepallength'],bins=100)
plt.subplot(232)
plt.hist(X_norm[:,0],bins=100)

# PCA训练
print('X的维数：',X.shape)
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_norm)

var_ratio = pca.explained_variance_ratio_
print('比例：',var_ratio)

plt.subplot(233)
plt.bar([1,2,3,4],var_ratio)
plt.xticks([1,2,3,4],['PC1','PC2','PC3','PC4'])
plt.ylabel('variance ratio')

# 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)

plt.subplot(235)
setosa = plt.scatter(X_pca[:,0][y=='Iris-setosa'],X_pca[:,1][y=='Iris-setosa'])
versicolor = plt.scatter(X_pca[:,0][y=='Iris-versicolor'],X_pca[:,1][y=='Iris-versicolor'])
virginica = plt.scatter(X_pca[:,0][y=='Iris-virginica'],X_pca[:,1][y=='Iris-virginica'])
plt.legend((setosa,versicolor,virginica),('setosa','versicolor','virginica'))

# 模型评估
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_pca,y)
y_predict = KNN.predict(X_pca)
accuracy = accuracy_score(y,y_predict)
print('KNN准确率：\n',accuracy)

plt.subplot(236)
setosa = plt.scatter(X_pca[:,0][y_predict=='Iris-setosa'],X_pca[:,1][y_predict=='Iris-setosa'])
versicolor = plt.scatter(X_pca[:,0][y_predict=='Iris-versicolor'],X_pca[:,1][y_predict=='Iris-versicolor'])
virginica = plt.scatter(X_pca[:,0][y_predict=='Iris-virginica'],X_pca[:,1][y_predict=='Iris-virginica'])
plt.legend((setosa,versicolor,virginica),('setosa','versicolor','virginica'))
plt.show()