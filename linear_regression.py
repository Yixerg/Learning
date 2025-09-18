# 加载数据
import pandas as pd
data = pd.read_csv('data_regression.csv')

print(data.head())  #查看数据前几行
print(type(data),data.shape)    #打印类型与形状,data类型是'pandas.core.frame.DataFrame'

# data赋值
x = data.loc[:,'x'] #x的类型是'pandas.core.series.Series'
y = data.loc[:,'y']
print(x,y)
print(type(x),x.shape,type(y),y.shape)

# 数据可视化
from matplotlib import pyplot as plt
plt.figure()
plt.scatter(x,y)
plt.title('1')
plt.show()

# 建立线性回归模型
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()

# 把x，y弄成10行1列矩阵
import numpy as np
x = np.array(x)
x = x.reshape(-1,1)
y = np.array(y)
y = y.reshape(-1,1)
print(type(x),x.shape,type(y),y.shape)

# 训练
lr_model.fit(x,y) #要将x,y转化为多维数组，不然会报错
print('训练完成')

# 预测数据
y_predict = lr_model.predict(x)
print(y_predict,type(y))
print(y,type(y))

y_3 = lr_model.predict([[3.5]])
print(y_3,type(y_3))

# 打印a，b
a = lr_model.coef_
b = lr_model.intercept_
print('a=',a,'b=',b)

# 用MSE、R**2方法检验
from sklearn.metrics import mean_squared_error,r2_score
MSE = mean_squared_error(y,y_predict)
R2 = r2_score(y,y_predict)
print(MSE,R2)

plt.figure()
plt.scatter(y,y_predict)
plt.title('2')
plt.show()