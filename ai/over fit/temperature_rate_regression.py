# 建立线性回归模型，计算r2分数，可视化预测结果
# 加入多项式特征
# 计算多项式特征的r2分数
# 预测结果

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# 加载数据
data_train = pd.read_csv('D:/ai/over fit/T-R-train.csv')
print('检查数据：\n',data_train.head())

# 赋值
X_train = data_train.loc[:,'T']
y_train = data_train.loc[:,'rate']

plt.figure(figsize=(18,12))
plt.subplot(221)
plt.scatter(X_train,y_train)
plt.title('raw data')
plt.xlabel('temperature')
plt.ylabel('rate')

# 线性回归预测
from sklearn.linear_model import LinearRegression
lr1 = LinearRegression()
X_train = np.array(X_train).reshape(-1,1)
lr1.fit(X_train,y_train)

# 加载数据
data_train = pd.read_csv('D:/ai/confusion_matrix/T-R-test.csv')
X_test = data_train.loc[:,'T']
y_test = data_train.loc[:,'rate']
X_test = np.array(X_test).reshape(-1,1)

# 在训练数据与测试数据进行预测
y_train_predict = lr1.predict(X_train)
y_test_predict = lr1.predict(X_test)

from sklearn.metrics import r2_score
r2_train = r2_score(y_train,y_train_predict)
r2_test = r2_score(y_test,y_test_predict)
print('训练数据集r2：',r2_train)
print('测试数据集r2：',r2_test)

# 模型可视化
# 生成新的数据点
X_range = np.linspace(40,90,300).reshape(-1,1)
y_range_predict = lr1.predict(X_range)
plt.subplot(222)
plt.plot(X_range,y_range_predict)
plt.scatter(X_train,y_train)
plt.xlabel('temperature')
plt.ylabel('rate')
plt.title('poly1')

# 多项式模型
from sklearn.preprocessing import PolynomialFeatures
poly2 = PolynomialFeatures(degree=2)
X_2_train = poly2.fit_transform(X_train)
X_2_test = poly2.fit_transform(X_test)
print('预处理后数据：\n',X_2_train)

lr2 = LinearRegression()
lr2.fit(X_2_train,y_train)

y_2_train_predict = lr2.predict(X_2_train)
y_2_test_predict = lr2.predict(X_2_test)

r2_2_train = r2_score(y_train,y_2_train_predict)
r2_2_test = r2_score(y_test,y_2_test_predict)
print('训练数据集r2：',r2_2_train)
print('测试数据集r2：',r2_2_test)

X_2_range = poly2.fit_transform(X_range)
y_2_range_predict = lr2.predict(X_2_range)
plt.subplot(223)
plt.plot(X_range,y_2_range_predict)
plt.scatter(X_train,y_train)
plt.scatter(X_test,y_test)
plt.xlabel('temperature')
plt.ylabel('rate')
plt.title('poly2')

# 过拟合
poly5 = PolynomialFeatures(degree=5)
X_5_train = poly5.fit_transform(X_train)
X_5_test = poly5.fit_transform(X_test)
print('预处理后数据：\n',X_5_train)

lr5 = LinearRegression()
lr5.fit(X_5_train,y_train)

y_5_train_predict = lr5.predict(X_5_train)
y_5_test_predict = lr5.predict(X_5_test)

r2_5_train = r2_score(y_train,y_5_train_predict)
r2_5_test = r2_score(y_test,y_5_test_predict)
print('训练数据集r2：',r2_5_train)
print('测试数据集r2：',r2_5_test)

X_5_range = poly5.fit_transform(X_range)
y_5_range_predict = lr5.predict(X_5_range)
plt.subplot(224)
plt.plot(X_range,y_5_range_predict)
plt.scatter(X_train,y_train)
plt.scatter(X_test,y_test)
plt.xlabel('temperature')
plt.ylabel('rate')
plt.title('poly5')
plt.show()