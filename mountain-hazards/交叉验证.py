
# coding: utf-8

# In[1]:

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
def cross_validation():
    df = pd.read_excel(r'E:\typhoon\经济\经济全整理(1).xlsx')
    df1 = pd.read_excel(r'E:\typhoon\代码\train_validation.xlsx')
    df2 = pd.read_excel(r'E:\typhoon\代码\true.xlsx')
    x = df1
    y = df2.iloc[0:51,2]
    print(y.shape)
    y = np.reshape(y, (-1, 1))
    # #将数据分成训练集，验证集和测试集（此处无验证集）
    train_x_disorder, val_x_disorder,train_y_disorder , val_y_disorder = train_test_split(x, y,
                                                                                            train_size=0.6, random_state=33)
    return train_x_disorder, val_x_disorder,train_y_disorder , val_y_disorder

