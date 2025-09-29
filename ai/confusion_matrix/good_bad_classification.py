import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore') 

# 加载数据并且可视化
data = pd.read_csv('D:/ai/confusion_matrix/data_class_raw.csv')
print('数据检查：\n',data.head())

X = data.drop(['y'],axis=1)
y = data.loc[:,'y']

plt.figure(figsize=(18,12))
plt.subplot(221)
bad = plt.scatter(X.loc[:,'x1'][y==0],X.loc[:,'x2'][y==0])
good = plt.scatter(X.loc[:,'x1'][y==1],X.loc[:,'x2'][y==1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('raw data')
plt.legend((good,bad),('good','bad'))
# plt.show()

# 异常检测
from sklearn.covariance import EllipticEnvelope
ad_model = EllipticEnvelope(contamination=0.02)
ad_model.fit(X[y==0])
y_predict_bad = ad_model.predict(X[y==0])
print('异常点：',y_predict_bad)

plt.subplot(222)
bad = plt.scatter(X.loc[:,'x1'][y==0],X.loc[:,'x2'][y==0])
plt.scatter(X.loc[:,'x1'][y==0][y_predict_bad==-1],X.loc[:,'x2'][y==0][y_predict_bad==-1],marker='x',s=150)
good = plt.scatter(X.loc[:,'x1'][y==1],X.loc[:,'x2'][y==1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('raw data')
plt.legend((good,bad),('good','bad'))
# plt.show()

# 异常点检测完毕，对处理完的数据进行PCA处理
data = pd.read_csv('D:/ai/confusion_matrix/data_class_processed.csv')
print('数据检查：\n',data.head())

X = data.drop(['y'],axis=1)
y = data.loc[:,'y']

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_norm = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)


X_reduce = pca.fit_transform(X_norm)
var_ratio = pca.explained_variance_ratio_
print('占比：',var_ratio)

plt.subplot(223)
plt.bar([1,2],var_ratio)
# plt.show()

# 训练集与测试集分离
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=4,test_size=0.4)
print('数据大小：',X_train.shape,X_test.shape,X.shape)

# 用KNN模型完成分类
from sklearn.neighbors import KNeighborsClassifier
knn_10 = KNeighborsClassifier(n_neighbors=10)
knn_10.fit(X_train,y_train)
y_train_predict = knn_10.predict(X_train)
y_test_predict = knn_10.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(y_train,y_train_predict)
accuracy_test = accuracy_score(y_test,y_test_predict)
print('训练集、测试集准确率：',accuracy_train,accuracy_test)

# 可视化分类边界
xx,yy = np.meshgrid(np.arange(0,10,0.05),np.arange(0,10,0.05))
x_range = np.c_[xx.ravel(),yy.ravel()]

y_range_predict = knn_10.predict(x_range)

plt.subplot(224)
knn_bad = plt.scatter(x_range[:,0][y_range_predict==0],x_range[:,1][y_range_predict==0])
knn_good = plt.scatter(x_range[:,0][y_range_predict==1],x_range[:,1][y_range_predict==1])
bad = plt.scatter(X.loc[:,'x1'][y==0],X.loc[:,'x2'][y==0],s=15)
good = plt.scatter(X.loc[:,'x1'][y==1],X.loc[:,'x2'][y==1],s=15)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend((good,bad,knn_good,knn_bad),('good','bad','knn_good','knn_bad'))
plt.show()

# 计算混淆矩阵，准确率、召回率、特异度、精确率、F1分数
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_test_predict)
print('混淆矩阵：\n',cm)

TP = cm[1,1]
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]

accuracy = (TP + TN)/(TP + TN + FP + FN)
print('准确率：',accuracy)

recall = TP/(TP + FN)
print('召回率：',recall)

specificity = TN/(TN + FP)
print('特异度：',specificity)

precision = TP/(TP + FP)
print('精确率：',precision)

f1 = 2*precision*recall/(precision + recall)
print('F1分数：',f1)

# 改变n_neighbor，计算准确率
n = [i for i in range(1,21)]
accuracy_train = []
accuracy_test = []

plt.figure(figsize=(18,12))

for i in n:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_train_predict = knn.predict(X_train)
    y_test_predict = knn.predict(X_test)
    accuracy_score_train = accuracy_score(y_train,y_train_predict)
    accuracy_score_test = accuracy_score(y_test,y_test_predict)
    accuracy_train.append(accuracy_score_train)
    accuracy_test.append(accuracy_score_test)
    y_range_predict = knn.predict(x_range)
    plt.subplot(4,5,i)
    knn_bad = plt.scatter(x_range[:,0][y_range_predict==0],x_range[:,1][y_range_predict==0])
    knn_good = plt.scatter(x_range[:,0][y_range_predict==1],x_range[:,1][y_range_predict==1])
    bad = plt.scatter(X.loc[:,'x1'][y==0],X.loc[:,'x2'][y==0],s=15)
    good = plt.scatter(X.loc[:,'x1'][y==1],X.loc[:,'x2'][y==1],s=15)
    plt.title('kneighbors=%d' % i)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend((good,bad,knn_good,knn_bad),('good','bad','knn_good','knn_bad'))
plt.show()

print('训练集准确度：\n',accuracy_train)
print('测试集准确度：\n',accuracy_test)

# for i in range(0,20):
#     print(accuracy_train[i],accuracy_test[i],'\n')

plt.figure(figsize=(18,12))
plt.subplot(131)
plt.scatter(accuracy_train[:],accuracy_test[:])
plt.xlabel('accuracy_train')
plt.ylabel('accuracy_test')
plt.title('kneighbors_visual')
# plt.show()

plt.subplot(132)
plt.plot(n,accuracy_train,marker='o')
plt.subplot(133)
plt.plot(n,accuracy_test,marker='o')
plt.show()