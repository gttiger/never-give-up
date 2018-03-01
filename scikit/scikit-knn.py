import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd

# df1 = pd.read_excel(r'E:\山地\排序\10.xlsx')
# df2 = pd.read_excel(r'E:\山地\整理第一把\标签.xlsx')
# df5 =pd.read_excel(r'E:\山地\整理第一把\标签测试.xlsx')
# df4 =  pd.read_excel(r'E:\山地\排序1\10.xlsx')
# df3 = pd.read_excel(r'E:\山地\整理第一把\四标签13.xlsx')
# df6 =pd.read_excel(r'E:\山地\整理第一把\四标签13测试.xlsx')


df1 = pd.read_excel(r'E:\山地\整理第一把\对比.xlsx')
df2 = pd.read_excel(r'E:\山地\整理第一把\对比单标签.xlsx')
df5 =pd.read_excel(r'E:\山地\整理第一把\对比单测试标签.xlsx')
df4 =  pd.read_excel(r'E:\山地\整理第一把\对比测试.xlsx')
df3 = pd.read_excel(r'E:\山地\整理第一把\四标签13.xlsx')
df6 =pd.read_excel(r'E:\山地\整理第一把\四标签13测试.xlsx')
min_max_scaler_x = preprocessing.MinMaxScaler()
min_max_scaler_y = preprocessing.MinMaxScaler()
min_max_scaler_z = preprocessing.MinMaxScaler()
feature_train_scaled = df1
feature_test_scaled = df4
# feature_train_scaled = min_max_scaler_x.fit_transform(df1)
# feature_test_scaled = min_max_scaler_x.fit_transform(df4)
label0_scaled = df2
label1_scaled = df5
label2_scaled = df3
label3_scaled = df6

def cross_validation(trainset,label):
    # #将数据分成训练集，验证集和测试集（此处无验证集）
    train_x_disorder, val_x_disorder,train_y_disorder , val_y_disorder = train_test_split(trainset, label,
                                                                                            train_size=0.75,random_state=33)
    return train_x_disorder, val_x_disorder,train_y_disorder , val_y_disorder

train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
                                                                                              label0_scaled)

from sklearn import neighbors
KNN = neighbors.KNeighborsClassifier(n_neighbors = 3)
KNN.fit(train_x_disorder,train_y_disorder)
print("KNN's.score:",KNN.score(feature_test_scaled,label1_scaled))