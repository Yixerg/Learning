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
fig1 = plt.subplot(2,4,1)
plt.scatter(X.loc[:,'V1'],X.loc[:,'V2'])
plt.title('unlabeled data')
plt.xlabel('V1')
plt.ylabel('V2')

fig2 = plt.subplot(2,4,2)
label0 = plt.scatter(X.loc[:,'V1'][y==0],X.loc[:,'V2'][y==0])
label1 = plt.scatter(X.loc[:,'V1'][y==1],X.loc[:,'V2'][y==1])
label2 = plt.scatter(X.loc[:,'V1'][y==2],X.loc[:,'V2'][y==2])
plt.title('labeled data')
plt.legend((label0,label1,label2),('label0','label1','label2'))

# 建立模型
from sklearn.cluster import KMeans
KM = KMeans(n_clusters=3,random_state=0)
KM.fit(X)

# 查看聚类中心
centers = KM.cluster_centers_

fig3 = plt.subplot(2,4,3)
label0 = plt.scatter(X.loc[:,'V1'][y==0],X.loc[:,'V2'][y==0])
label1 = plt.scatter(X.loc[:,'V1'][y==1],X.loc[:,'V2'][y==1])
label2 = plt.scatter(X.loc[:,'V1'][y==2],X.loc[:,'V2'][y==2])
plt.title('center')
plt.legend((label0,label1,label2),('label0','label1','label2'))
plt.scatter(centers[:,0],centers[:,1])

# 预测V1=80，V2=60的数据类别
y_predict_test = KM.predict([[80,60]])
print(y_predict_test)

# 基于训练预测数据
y_predict = KM.predict(X)
print('预测之后的分类个数：')
print(pd.Series(y_predict).value_counts())

# 查看准确率
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y,y_predict)
print('准确率：')
print(accuracy)

fig4 = plt.subplot(2,4,5)
label0 = plt.scatter(X.loc[:,'V1'][y==0],X.loc[:,'V2'][y==0])
label1 = plt.scatter(X.loc[:,'V1'][y==1],X.loc[:,'V2'][y==1])
label2 = plt.scatter(X.loc[:,'V1'][y==2],X.loc[:,'V2'][y==2])
plt.title('original')
plt.legend((label0,label1,label2),('label0','label1','label2'))
plt.scatter(centers[:,0],centers[:,1])

fig5 = plt.subplot(2,4,6)
label0 = plt.scatter(X.loc[:,'V1'][y_predict==0],X.loc[:,'V2'][y_predict==0])
label1 = plt.scatter(X.loc[:,'V1'][y_predict==1],X.loc[:,'V2'][y_predict==1])
label2 = plt.scatter(X.loc[:,'V1'][y_predict==2],X.loc[:,'V2'][y_predict==2])
plt.title('now')
plt.legend((label0,label1,label2),('label0','label1','label2'))
plt.scatter(centers[:,0],centers[:,1])

# 类别矫正
y_correct = []
for i in y_predict:
    if i == 0:
        y_correct.append(2)
    if i == 2:
        y_correct.append(1)
    if i == 1:
        y_correct.append(0)
print('改正后个数：')
print(pd.Series(y_correct).value_counts())
print('改正后准确率：')
print(accuracy_score(y,y_correct))

y_correct = np.array(y_correct)

fig5 = plt.subplot(2,4,7)
label0 = plt.scatter(X.loc[:,'V1'][y==0],X.loc[:,'V2'][y==0])
label1 = plt.scatter(X.loc[:,'V1'][y==1],X.loc[:,'V2'][y==1])
label2 = plt.scatter(X.loc[:,'V1'][y==2],X.loc[:,'V2'][y==2])
plt.title('original')
plt.legend((label0,label1,label2),('label0','label1','label2'))
plt.scatter(centers[:,0],centers[:,1])

fig6 = plt.subplot(2,4,8)
label0 = plt.scatter(X.loc[:,'V1'][y_correct==0],X.loc[:,'V2'][y_correct==0])
label1 = plt.scatter(X.loc[:,'V1'][y_correct==1],X.loc[:,'V2'][y_correct==1])
label2 = plt.scatter(X.loc[:,'V1'][y_correct==2],X.loc[:,'V2'][y_correct==2])
plt.title('correct')
plt.legend((label0,label1,label2),('label0','label1','label2'))
plt.scatter(centers[:,0],centers[:,1])

plt.show()