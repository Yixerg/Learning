# 加载数据
import pandas as pd
import numpy as np
data = pd.read_csv('examdata.csv')
print('数据前几行：')
print(data.head())

# 数据可视化
from matplotlib import pyplot as plt
fig1 = plt.figure()
plt.scatter(data.loc[:,'exam1'],data.loc[:,'exam2'])
plt.title('exam1-exam2')
plt.xlabel('exam1')
plt.ylabel('exam2')
plt.show()

# 数据分类显示
mask = data.loc[:,'pass'] == 1
print('是否通过：')
print(mask)
fig2 = plt.figure()
passed = plt.scatter(data.loc[:,'exam1'][mask],data.loc[:,'exam2'][mask])
failed = plt.scatter(data.loc[:,'exam1'][~mask],data.loc[:,'exam2'][~mask])
plt.title('exam1-exam2 divided')
plt.xlabel('exam1')
plt.ylabel('exam2')
plt.legend((passed,failed),('passed','failed'))
plt.show()

# 赋值
x = data.drop(['pass'],axis = 1)
y = data.loc[:,'pass']
x1 = data.loc[:,'exam1']
x2 = data.loc[:,'exam2']
print('查看x：')
print(x.head())

# 建立训练模型
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x,y)

# 预测结果与评估模型表现
y_predict = lr.predict(x)
print('预测是否通过：')
print(y_predict)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y,y_predict)
print('预测准确率：')
print(accuracy)

# 预测特定结果
import warnings
warnings.filterwarnings('ignore')  # 忽略所有警告

y_test = lr.predict([[70,65]])

print('70，65是否通过：')
print('passed' if y_test == 1 else 'failed')

# 结果可视化
theta0 = lr.intercept_
theta1,theta2 = lr.coef_[0][0],lr.coef_[0][1]
print('theta0:',theta0,'    theta1:',theta1,'   theta2:',theta2)
x2_new = -(theta0 + theta1*x1)/theta2

fig3 = plt.figure()
plt.plot(x1,x2_new)
passed = plt.scatter(data.loc[:,'exam1'][mask],data.loc[:,'exam2'][mask])
failed = plt.scatter(data.loc[:,'exam1'][~mask],data.loc[:,'exam2'][~mask])
plt.title('exam1-exam2 divided2')
plt.xlabel('exam1')
plt.ylabel('exam2')
plt.legend((passed,failed),('passed','failed'))
plt.show()

# 以上是一阶的训练，以下为二阶训练，引入x1^2,x2^2,x1*x2
# 创建新的数据集
x1_2 = x1*x1
x2_2 = x2*x2
x1_x2 = x1*x2

X_new = {'x1':x1 , 'x2':x2 , 'x1_2':x1_2 , 'x2_2':x2_2 , 'x1_x2':x1_x2}
X_new = pd.DataFrame(X_new)
print('新的数据集：')
print(X_new)

# 训练新的模型
lr2 = LogisticRegression()
lr2.fit(X_new,y)

# 预测结果与评估模型表现
y_predict_2 = lr2.predict(X_new)
accuracy_2 = accuracy_score(y,y_predict_2)
print(accuracy_2)

# 结果可视化
theta0 = lr2.intercept_
theta1,theta2,theta3,theta4,theta5 = lr2.coef_[0][0],lr2.coef_[0][1],lr2.coef_[0][2],lr2.coef_[0][3],lr2.coef_[0][4]
print('theta0:',theta0,'    theta1:',theta1,'   theta2:',theta2,'    theta3:',theta3,'   theta4:',theta4,'   theta5:',theta5)
x1_new = x1.sort_values()
a = theta4
b = theta5*x1_new + theta2
c = theta0 + theta1*x1_new + theta3*x1_new*x1_new
x2_new_boundary = np.where((-b+np.sqrt(b*b-4*a*c))/(2*a)>=0 , (-b+np.sqrt(b*b-4*a*c))/(2*a) , np.nan)
print('x2_new_boundary:\n',x2_new_boundary)

fig4 = plt.figure()
plt.plot(x1_new,x2_new_boundary)
passed = plt.scatter(data.loc[:,'exam1'][mask],data.loc[:,'exam2'][mask])
failed = plt.scatter(data.loc[:,'exam1'][~mask],data.loc[:,'exam2'][~mask])
plt.title('exam1-exam2 divided3')
plt.xlabel('exam1')
plt.ylabel('exam2')
plt.legend((passed,failed),('passed','failed'))
plt.show()