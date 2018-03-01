
# coding: utf-8

# In[16]:

import pandas as pd
import numpy as np


# In[17]:

df = pd.read_excel(r'E:\山地\泥石流(1).xlsx')


# In[18]:

import tensorflow as tf


# In[19]:

x = df.iloc[0:100,[0,2]]


# In[37]:

y = df.iloc[0:100,-12]
#y = np.where([y == '低易发',y == '中易发'],[-1,1],0)
y= y.replace(['低易发','中易发','高易发','不易发'],[0,1,2,3])

