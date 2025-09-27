import numpy as np
import pandas as pd

# 定义参数
num_points = 300  # 点的数量

# x1参数：区间5-12.5，中心为8.75
x1_mean = (5 + 12.5) / 2  # 8.75
x1_std = (12.5 - 5) / 6    # 区间宽度的1/6作为标准差，确保大部分点落在区间内

# x2参数：区间7.5-13，中心为10.25
x2_mean = (7.5 + 13) / 2   # 10.25
x2_std = (13 - 7.5) / 6    # 区间宽度的1/6作为标准差

# 生成符合正态分布的点
np.random.seed(42)  # 设置随机种子，确保结果可复现
x1 = np.random.normal(loc=x1_mean, scale=x1_std, size=num_points)
x2 = np.random.normal(loc=x2_mean, scale=x2_std, size=num_points)

# 确保所有点都在指定区间内（处理可能的异常值）
x1 = np.clip(x1, 5, 12.5)
x2 = np.clip(x2, 7.5, 13)

# 创建DataFrame并导出为CSV
df = pd.DataFrame({'x1': x1, 'x2': x2})
df.to_csv('addata.csv', index=False)

print(f"已生成{num_points}个点并保存为addata.csv")
print("数据示例：")
print(df.head())

from matplotlib import pyplot as plt
plt.figure()
plt.scatter(x1,x2)
plt.show()