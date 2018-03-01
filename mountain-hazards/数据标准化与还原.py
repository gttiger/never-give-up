
# coding: utf-8

# In[7]:

import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import LeaveOneOut


# In[8]:

df = pd.read_excel(r'E:\typhoon\经济\经济全整理(1).xlsx')
df1 = pd.read_excel(r'E:\typhoon\代码\train_validation(ann).xlsx')
df2 = pd.read_excel(r'E:\typhoon\代码\true.xlsx')


# In[9]:

ss_x = preprocessing.StandardScaler()
train_x = ss_x.fit_transform(df1)
val_x = ss_x.fit_transform(val_x_disorder)
ss_y = preprocessing.StandardScaler()
train_y = ss_y.fit_transform(df2)
val_y = ss_y.fit_transform( val_y_disorder)


# In[12]:

prediction1 = ss_x.inverse_transform(train_x)

