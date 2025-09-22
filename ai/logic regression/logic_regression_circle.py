# 加载数据
import pandas as pd
import numpy as np
data = pd.read_csv('chip_test.csv')
print('数据前几行：')
print(data.head())

# 数据可视化
from matplotlib import pyplot as plt
plt.figure(figsize=(18,12))

fig1 = plt.subplot(2,3,1)
plt.scatter(data.loc[:,'test1'],data.loc[:,'test2'])
plt.title('test1-test2')
plt.xlabel('test1')
plt.ylabel('test2')

# 数据分类显示
mask = data.loc[:,'pass'] == 1
print('是否通过：')
print(mask)
fig2 = plt.subplot(2,3,2)
passed = plt.scatter(data.loc[:,'test1'][mask],data.loc[:,'test2'][mask])
failed = plt.scatter(data.loc[:,'test1'][~mask],data.loc[:,'test2'][~mask])
plt.title('test1-test2 divided')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed,failed),('passed','failed'))

# 赋值
x = data.drop(['pass'],axis = 1)
y = data.loc[:,'pass']
x1 = data.loc[:,'test1']
x2 = data.loc[:,'test2']
print('查看x：')
print(x.head())

# 创建新的数据集
x1_2 = x1*x1
x2_2 = x2*x2
x1_x2 = x1*x2

X_new = {'x1':x1 , 'x2':x2 , 'x1_2':x1_2 , 'x2_2':x2_2 , 'x1_x2':x1_x2}
X_new = pd.DataFrame(X_new)
print('新的数据集：')
print(X_new)

import warnings
warnings.filterwarnings('ignore')  # 忽略所有警告

# 建立训练模型
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_new,y)

# 预测结果与评估模型表现
from sklearn.metrics import accuracy_score
y_predict = lr.predict(X_new)
accuracy = accuracy_score(y,y_predict)
print(accuracy)

# 结果可视化
theta0 = lr.intercept_
theta1,theta2,theta3,theta4,theta5 = lr.coef_[0][0],lr.coef_[0][1],lr.coef_[0][2],lr.coef_[0][3],lr.coef_[0][4]
print('theta0:',theta0,'    theta1:',theta1,'   theta2:',theta2,'    theta3:',theta3,'   theta4:',theta4,'   theta5:',theta5)
x1_new = x1.sort_values()
a = theta4
b = theta5*x1_new + theta2
c = theta0 + theta1*x1_new + theta3*x1_new*x1_new
# x2_new_boundary = np.where((-b+np.sqrt(b*b-4*a*c))/(2*a)>=0 , (-b+np.sqrt(b*b-4*a*c))/(2*a) , np.nan)
x2_new_boundary = (-b+np.sqrt(b*b-4*a*c))/(2*a)
print('x2_new_boundary:\n',x2_new_boundary)

fig3 = plt.subplot(2,3,3)
plt.plot(x1_new,x2_new_boundary)
passed = plt.scatter(data.loc[:,'test1'][mask],data.loc[:,'test2'][mask])
failed = plt.scatter(data.loc[:,'test1'][~mask],data.loc[:,'test2'][~mask])
plt.title('test1-test2 after learning')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed,failed),('passed','failed'))

# 定义函数求解
def f(x):
    a = theta4
    b = theta5*x + theta2
    c = theta0 + theta1*x + theta3*x*x
    x2_new_boundary1 = np.where(b*b-4*a*c>=0 , (-b+np.sqrt(b*b-4*a*c))/(2*a) , np.nan)
    x2_new_boundary2 = np.where(b*b-4*a*c>=0 , (-b-np.sqrt(b*b-4*a*c))/(2*a) , np.nan)
    # x2_new_boundary1 = (-b+np.sqrt(b*b-4*a*c))/(2*a)
    # x2_new_boundary2 = (-b-np.sqrt(b*b-4*a*c))/(2*a)
    return x2_new_boundary1,x2_new_boundary2

# x2_new_boundary1 = []
# x2_new_boundary2 = []
# for x in x1_new:
#     x2_new_boundary1.append(f(x)[0])
#     x2_new_boundary2.append(f(x)[1])
# print('函数返回结果：')
# print(x2_new_boundary1,x2_new_boundary2)
x2_new_boundary1, x2_new_boundary2 = f(x1_new)

fig4 = plt.subplot(2,3,4)
plt.plot(x1_new,x2_new_boundary1)
plt.plot(x1_new,x2_new_boundary2)
passed = plt.scatter(data.loc[:,'test1'][mask],data.loc[:,'test2'][mask])
failed = plt.scatter(data.loc[:,'test1'][~mask],data.loc[:,'test2'][~mask])
plt.title('test1-test2 two lines')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed,failed),('passed','failed'))

# 生成数据集，使曲线完整
x1_range = [-1 + x/10000 for x in range(0,20000)]
x1_range = np.array(x1_range)
x2_new_boundary1, x2_new_boundary2 = f(x1_range)

fig5 = plt.subplot(2,3,5)
plt.plot(x1_range,x2_new_boundary1)
plt.plot(x1_range,x2_new_boundary2)
passed = plt.scatter(data.loc[:,'test1'][mask],data.loc[:,'test2'][mask])
failed = plt.scatter(data.loc[:,'test1'][~mask],data.loc[:,'test2'][~mask])
plt.title('test1-test2 final')
plt.xlabel('test1')
plt.ylabel('test2')
plt.legend((passed,failed),('passed','failed'))
plt.show()