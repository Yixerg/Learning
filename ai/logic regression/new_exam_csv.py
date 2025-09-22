import numpy as np
import pandas as pd

np.random.seed(42)

n_rows = 100

exam1 = np.random.randint(0,101,n_rows)
exam2 = np.random.randint(0,101,n_rows)

pass_status = np.where((exam1 > 60) & (exam2 > 60) , 1 , 0)

data = pd.DataFrame({
    'exam1':exam1,
    'exam2':exam2,
    'pass':pass_status
})

data.to_csv('examdata.csv',index=False,encoding='utf-8')
print('成功')
