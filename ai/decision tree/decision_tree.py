# 加载数据
import pandas as pd
import numpy as np
data = pd.read_csv('d:/ai/decision tree/iris.csv')
print('查看导入数据：\n',data.head())

# 赋值
X = data.drop(['class'],axis=1)
y = data.loc[:,'class']
print('查看X赋值：\n',X.head())
print('查看y赋值：\n',y.head())

# 建立模型
from sklearn import tree
dc_tree = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=5)
dc_tree.fit(X,y)

# 可视化决策树
from matplotlib import pyplot as plt
plt.figure()
tree.plot_tree(dc_tree,filled=True,feature_names = ['sepallength','sepalwidth','petallength','petalwidth'],class_names = ['setosa','versicolor','virginica'])

# 模型评估
from sklearn.metrics import accuracy_score
y_predict = dc_tree.predict(X)
accuracy = accuracy_score(y,y_predict)
print('模型准确率：\n',accuracy)

sample = [[5.5,3.2,1.9,0.7]]
sample_df = pd.DataFrame(sample, columns=['sepallength', 'sepalwidth', 'petallength', 'petalwidth'])
prediction = dc_tree.predict(sample_df)
print('预测种类：',prediction)

plt.show()