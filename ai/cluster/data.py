import numpy as np
import pandas as pd

np.random.seed(42)

n_rows = 100

V11 = np.random.uniform(0,70,n_rows)
V21 = np.random.uniform(23,90,n_rows)
label1 = np.ones(100 , dtype=int)

V12 = np.random.uniform(-20,30,n_rows)
V22 = np.random.uniform(-20,30,n_rows)
label2 = np.full(100 , 2 , dtype=int)

V13 = np.random.uniform(40,100,n_rows)
V23 = np.random.uniform(-40,27,n_rows)
label3 = np.zeros(100 , dtype=int)

V1 = np.concatenate([V11,V12,V13])
V2 = np.concatenate([V21,V22,V23])
label = np.concatenate([label1,label2,label3])

data = pd.DataFrame({
    'V1':V1,
    'V2':V2,
    'label':label
})

data.to_csv('data.csv',index=False,encoding='utf-8')
print('成功')

from matplotlib import pyplot as plt
plt.figure()
plt.scatter(data[data['label'] == 0]['V1'], data[data['label'] == 0]['V2'], 
           alpha=0.7, label='Class 0', color='red')
plt.scatter(data[data['label'] == 1]['V1'], data[data['label'] == 1]['V2'], 
           alpha=0.7, label='Class 1', color='blue')
plt.scatter(data[data['label'] == 2]['V1'], data[data['label'] == 2]['V2'], 
           alpha=0.7, label='Class 2', color='green')

plt.legend()
plt.show()