
# coding: utf-8

# In[1]:

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing


# In[4]:

# df5 = pd.read_excel(r"E:\山地\泥石流规模\泥石流转化.xlsx")
# df2 = pd.read_excel(r"E:\山地\泥石流规模\test.xlsx")
#df = pd.read_excel(r"E:\山地\正规第三把\单标签13resample.xlsx",index = False) 
df = pd.read_excel(r"E:\山地\整理第一把\label131.xlsx",index = False) 
#y =df['泥石流规模等级'].replace('0','[0,0,0,1]')
#df1 = pd.read_excel(r"E:\山地\泥石流规模\1.xlsx",index = False)
#df2 = pd.read_excel(r"E:\山地\泥石流规模\label.xlsx")
#df3 = pd.read_excel(r"E:\山地\泥石流规模\2.xlsx")
# df3.iloc[:,0] = y
#df3 = pd.DataFrame(np.random.randn(3821,4),columns=['易发性极低','易发性中','易发性高','易发性极高'])
df3 = pd.DataFrame(np.random.randn(1778,4),columns=['不易发','低易发','中易发','高易发'])
list1 = [0,0,0,1]
list2 = [0,0,1,0]
list3 = [0,1,0,0]
list4 = [1,0,0,0]
#df_empty = pd.DataFrame()  
for i in range(len(df)):
    if df.iloc[i,0] ==0:
        df3.iloc[i,:] = list4
        #df_empty = df_empty.append(df1.iloc[0,0:3])
    elif df.iloc[i,0] ==1:
        df3.iloc[i,:] = list3
        #df_empty = df_empty.append(df1.iloc[1,0:3])
    elif df.iloc[i,0] ==2:
        df3.iloc[i,:] = list2
        #df_empty = df_empty.append(df1.iloc[2,0:3])
    elif df.iloc[i,0] ==3:
        df3.iloc[i,:] = list1
        #df_empty = df_empty.append(df1.iloc[3,0:3])
#df3.to_excel(r"E:\山地\正规第三把\四标签131resampe.xlsx",index = False)    
df3.to_excel(r"E:\山地\整理第四把\对比标签.xlsx",index = False)  
# min_max_scaler = preprocessing.MinMaxScaler()
# standar_scaler = preprocessing.StandardScaler()
# feature_scaled = min_max_scaler.fit_transform(df1)
# #feature_3_scaled = min_max_scaler.fit_transform(df2)
# label_scaled = min_max_scaler.fit_transform(df2)


# In[6]:

#归一化
train_set = df1.iloc[0:1760,]
min_max_scaler = preprocessing.MinMaxScaler()
standar_scaler = preprocessing.StandardScaler()
feature_1_scaled = standar_scaler.fit_transform(df5)
feature_3_scaled = min_max_scaler.fit_transform(df2)
label_scaled = min_max_scaler.fit_transform(df4)

