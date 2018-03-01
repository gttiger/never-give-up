
# coding: utf-8

# In[17]:

from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd


# In[41]:

df1 = pd.read_excel(r'E:\typhoon\代码\train_validation.xlsx')
df2 = pd.read_excel(r'E:\typhoon\代码\true.xlsx')
Y = df2.iloc[0:51,2]
#lol = cross_validation.LeaveOneLabelOut(df1)
loo = LeaveOneOut()


# In[43]:

for train_index,test_index in loo.split(df1):
    print("%s %s" %(train_index,test_index))
    x_train,x_test = df1.iloc[train_index,0:2],df1.iloc[test_index,0:2]
    y_train,y_test = Y.iloc[train_index],Y.iloc[test_index]
    print(x_train,x_test,y_train,y_test)


# In[27]:

from sklearn.model_selection import LeaveOneOut
X = np.array([[1, 2], [3, 4]])
y = np.array([1, 2])
loo = LeaveOneOut()
loo.get_n_splits(X)
for train_index, test_index in loo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

