import numpy as np
import pandas as pd
from pandas import DataFrame, Series

dic = {
    'age': [1, 0, 1],
    'weight': [99, 77, 33],
    'no': [55, 88, 65]
}
df1 = DataFrame(data=dic, index=['xiaoming', 'lingling', 'xiaoli'])

print(df1)
l = []
for index, item in enumerate(df1['age']):
    # print(index,item)
    if item == 1:
        l.append(10)
    else:
        l.append(0)
df1['age2']=l

print(df1)