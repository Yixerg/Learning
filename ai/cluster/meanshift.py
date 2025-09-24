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

# 建立模型
from sklearn.cluster import MeanShift,estimate_bandwidth

# 估算带宽
bw = estimate_bandwidth(X,n_samples=500)
print('带宽：',bw)

ms = MeanShift(bandwidth=bw)
ms.fit(X)

# 查看结果类别
y_predict_ms = ms.predict(X)
print('类别：\n',pd.Series(y_predict_ms).value_counts())

from sklearn.metrics import accuracy_score

# 类别矫正
y_correct = []
for i in y_predict_ms:
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

# 结果可视化
fig5 = plt.subplot(2,2,3)
label0 = plt.scatter(X.loc[:,'V1'][y==0],X.loc[:,'V2'][y==0])
label1 = plt.scatter(X.loc[:,'V1'][y==1],X.loc[:,'V2'][y==1])
label2 = plt.scatter(X.loc[:,'V1'][y==2],X.loc[:,'V2'][y==2])
plt.title('original')
plt.legend((label0,label1,label2),('label0','label1','label2'))

fig6 = plt.subplot(2,2,4)
label0 = plt.scatter(X.loc[:,'V1'][y_correct==0],X.loc[:,'V2'][y_correct==0])
label1 = plt.scatter(X.loc[:,'V1'][y_correct==1],X.loc[:,'V2'][y_correct==1])
label2 = plt.scatter(X.loc[:,'V1'][y_correct==2],X.loc[:,'V2'][y_correct==2])
plt.title('correct')
plt.legend((label0,label1,label2),('label0','label1','label2'))

plt.show()