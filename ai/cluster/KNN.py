# 加载数据
import pandas as pd
import numpy as np
data = pd.read_csv('data.csv')
print('检查数据：')
print(data.head())

# 赋值
X = data.drop(['label'],axis=1)
y = data.loc[:,'label']
print('X检查：')
print(X.head())
print('y检查：')
print(y.head())

print('统计标签个数：')
print(pd.Series(y).value_counts())

# 可视化
from matplotlib import pyplot as plt

plt.figure(figsize=(18,12))

fig1 = plt.subplot(2,2,1)
plt.scatter(X.loc[:,'V1'],X.loc[:,'V2'])
plt.title('unlabeled data')
plt.xlabel('V1')
plt.ylabel('V2')

fig2 = plt.subplot(2,2,2)
label0 = plt.scatter(X.loc[:,'V1'][y==0],X.loc[:,'V2'][y==0])
label1 = plt.scatter(X.loc[:,'V1'][y==1],X.loc[:,'V2'][y==1])
label2 = plt.scatter(X.loc[:,'V1'][y==2],X.loc[:,'V2'][y==2])
plt.title('labeled data')
plt.legend((label0,label1,label2),('label0','label1','label2'))

# 建立KNN模型
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X,y)

# 预测V1=80，V2=60的数据类别
y_predict_knn_test = KNN.predict([[80,60]])
print('KNN预测结果：')
print(y_predict_knn_test)

# 模型准确率
from sklearn.metrics import accuracy_score
y_predict_knn = KNN.predict(X)
accuracy = accuracy_score(y,y_predict_knn)
print('准确率：',accuracy)

# 可视化
fig3 = plt.subplot(2,2,3)
label0 = plt.scatter(X.loc[:,'V1'][y==0],X.loc[:,'V2'][y==0])
label1 = plt.scatter(X.loc[:,'V1'][y==1],X.loc[:,'V2'][y==1])
label2 = plt.scatter(X.loc[:,'V1'][y==2],X.loc[:,'V2'][y==2])
plt.title('original')
plt.legend((label0,label1,label2),('label0','label1','label2'))

fig4 = plt.subplot(2,2,4)
label0 = plt.scatter(X.loc[:,'V1'][y_predict_knn==0],X.loc[:,'V2'][y_predict_knn==0])
label1 = plt.scatter(X.loc[:,'V1'][y_predict_knn==1],X.loc[:,'V2'][y_predict_knn==1])
label2 = plt.scatter(X.loc[:,'V1'][y_predict_knn==2],X.loc[:,'V2'][y_predict_knn==2])
plt.title('correct')
plt.legend((label0,label1,label2),('label0','label1','label2'))
plt.show()