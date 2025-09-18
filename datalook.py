import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('xy_test.csv')
print(data.head())

x = data.loc[:,'x'] 
y = data.loc[:,'y']

plt.figure()
plt.scatter(x,y)
plt.show()

x = np.array(x)
x = x.reshape(-1,1)
y = np.array(y)
y = y.reshape(-1,1)

from sklearn.linear_model import LinearRegression
foo = LinearRegression()

foo.fit(x,y)
print('成功')

a = foo.coef_
b = foo.intercept_
print('a=',a,'b=',b)

y_predict = foo.predict(x)

plt.figure()
plt.scatter(y,y_predict)
plt.show()

