# 加载数据
import numpy as np
import pandas as pd
data = pd.read_csv('d:/ai/anomaly detection/addata.csv')
print('数据前几行：\n',data.head())

# 数据可视化
x1 = data.loc[:,'x1']
x2 = data.loc[:,'x2']
from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(x1,x2)
plt.show()

# 数据分布统计
plt.figure()
fig2 = plt.subplot(1,2,1)
plt.hist(x1,bins=100)
plt.xlabel('x1')
plt.ylabel('counts')
fig3 = plt.subplot(1,2,2)
plt.hist(x2,bins=100)
plt.xlabel('x2')
plt.ylabel('counts')
plt.show()

# 计算高斯分布
x1_mean = x1.mean()
x1_sigma = x1.std()
x2_mean = x2.mean()
x2_sigma = x2.std()
print('x1_mean:',x1_mean,'x1_sigma:',x1_sigma)
print('x2_mean:',x2_mean,'x2_sigma:',x2_sigma)

from scipy.stats import norm
x1_range = np.linspace(0,20,300)
normal1 = norm.pdf(x1_range,x1_mean,x1_sigma)
x2_range = np.linspace(0,20,300)
normal2 = norm.pdf(x2_range,x2_mean,x2_sigma)

plt.figure()
fig4 = plt.subplot(1,2,1)
plt.plot(x1_range,normal1)
plt.xlabel('x1')
plt.ylabel('counts')
fig5 = plt.subplot(1,2,2)
plt.plot(x2_range,normal2)
plt.xlabel('x2')
plt.ylabel('counts')
plt.show()

# 建立模型，进行预测
from sklearn.covariance import EllipticEnvelope
ad_model = EllipticEnvelope()
ad_model.fit(data)

y_predict = ad_model.predict(data)
print('预测结果：\n',y_predict)
print('个数：\n',pd.Series(y_predict).value_counts())

fig6 = plt.figure()
plt.xlabel('x1')
plt.ylabel('x2')
original_data = plt.scatter(data.loc[:,'x1'][y_predict==1],data.loc[:,'x2'][y_predict==1],marker='x')
anomaly_data = plt.scatter(data.loc[:,'x1'][y_predict==-1],data.loc[:,'x2'][y_predict==-1],marker='o')
plt.legend((original_data,anomaly_data),('original_data','anomaly_data'))
plt.show()

# 修改参数，使结果更加合理
ad_model = EllipticEnvelope(contamination=0.02)
ad_model.fit(data)

y_predict = ad_model.predict(data)
print('预测结果：\n',y_predict)
print('个数：\n',pd.Series(y_predict).value_counts())

fig6 = plt.figure()
plt.xlabel('x1')
plt.ylabel('x2')
original_data = plt.scatter(data.loc[:,'x1'][y_predict==1],data.loc[:,'x2'][y_predict==1],marker='x')
anomaly_data = plt.scatter(data.loc[:,'x1'][y_predict==-1],data.loc[:,'x2'][y_predict==-1],marker='o')
plt.legend((original_data,anomaly_data),('original_data','anomaly_data'))
plt.show()