# 数据读取
import pandas as pd
import numpy as np
data = pd.read_csv('usa_housing_price.csv')
print(data.head())

# 数据可视化
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10,10))

fig1 = plt.subplot(231)
plt.scatter(data.loc[:,'Avg Area1'],data.loc[:,'Price'])
plt.title('Price VS 1')

fig2 = plt.subplot(232)
plt.scatter(data.loc[:,'Avg Area2'],data.loc[:,'Price'])
plt.title('Price VS 2')

fig3 = plt.subplot(233)
plt.scatter(data.loc[:,'Avg Area3'],data.loc[:,'Price'])
plt.title('Price VS 3')

fig4 = plt.subplot(234)
plt.scatter(data.loc[:,'Area Population'],data.loc[:,'Price'])
plt.title('Price VS Area Population')

fig5 = plt.subplot(235)
plt.scatter(data.loc[:,'size'],data.loc[:,'Price'])
plt.title('Price VS size')
plt.show()

# 单因子线性回归
x = data.loc[:,'size']
y = data.loc[:,'Price']

x = np.array(x)
x = x.reshape(-1,1)
y = np.array(y)
y = y.reshape(-1,1)

from sklearn.linear_model import LinearRegression
lr1 = LinearRegression()
lr1.fit(x,y)

# 预测数据
y_predict_1 = lr1.predict(x)
print(y_predict_1)

# 评估与可视化
from sklearn.metrics import mean_squared_error,r2_score
mse1 = mean_squared_error(y,y_predict_1)
r2s1 = r2_score(y,y_predict_1)
print(mse1,r2s1)

fig6 = plt.figure(figsize=(8,5))
plt.scatter(x,y)
plt.plot(x,y_predict_1)
plt.show()

# 多因子线性回归
x_multi = data.drop(['Price'],axis=1)

lrmulti = LinearRegression()
lrmulti.fit(x_multi,y)

# 预测数据
y_predict_multi = lrmulti.predict(x_multi)
print(y_predict_multi)

# 评估与可视化
mse_multi = mean_squared_error(y,y_predict_multi)
r2s_multi = r2_score(y,y_predict_multi)
print(mse_multi,r2s_multi)

fig7 = plt.figure(figsize=(8,5))
plt.scatter(y,y_predict_multi)
plt.show()

# 具体数据预测
X_test = [65000,5,5,30000,200]
X_test = np.array(X_test)
X_test = X_test.reshape(1,-1)
Y_test = lrmulti.predict(X_test)
print(Y_test)
