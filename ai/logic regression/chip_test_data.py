import numpy as np
import pandas as pd

np.random.seed(42)

n_rows = 100

test1 = np.random.uniform(-1,1,n_rows)
test2 = np.random.uniform(-1,1,n_rows)

pass_status = np.where(test1**2 + test2**2 >= 0.36 , 1 , 0)

data = pd.DataFrame({
    'test1':test1,
    'test2':test2,
    'pass':pass_status
})

data.to_csv('chip_test.csv',index=False,encoding='utf-8')
print('成功')