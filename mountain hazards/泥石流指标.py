
# coding: utf-8

# In[6]:

import pandas as pd
import tensorflow as tf
import numpy as np


# In[7]:

df = pd.read_excel(r'E:\山地\dem\泥石流2.xlsx')


# In[20]:

def creation_data():
    #y =df['泥石流规模等级'].replace(['小型','中型','大型','巨型'],[[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
    y =df['易发程度'].replace(['不易发','$不易发$','低易发','$低易发$','中易发','$中等$','$中易发$','高易发','$高易发$','$易发$',],[0,0,1,1,2,2,2,3,3,3])
    
    x1 = df['不良地质现象'].replace(['无崩塌滑坡及冲沟或发育轻微','有零星崩塌滑坡和冲沟存在','崩塌滑坡发育,有零星植被覆盖,冲沟发育','崩塌滑坡严重,表土疏松,冲沟十分发育'],[1,2,3,4])
    x2 = df['植被覆盖率'].replace(['<10%','10-30%','30-60%','>60%'],[1,2,3,4])
    x3 = df['岩性因素'].replace(['软岩,黄土','软硬相间','风化和节理发育的硬岩','硬岩'],[1,2,3,4])
    x4 = df['松散物储量'].replace(['<1 万立方米/平方公里','5-1 万立方米/平方公里','10-5 万立方米/平方公里','>10 万立方米/平方公里'],[1,2,3,4])
    x5 = df['山坡坡度'].replace(['<15°(<268‰)','25-15°(466-268‰)','32-25°(625-466‰)','>32°(>625‰)'],[1,2,3,4])
    x6 = df['流域面积'].replace(['0.2 以下或10-100 平方公里','0.2-5 平方公里','<5 平方公里','5-10 平方公里','10-100 平方公里','>100 平方公里'],[1,2,3,4,5,6])
    x7 = df['相对高差'].replace(['<100m','300-100m','500-300m','>500m'],[1,2,3,4])
    x8= df['冲淤变幅'].replace(['<0.2m','1-0.2m','2-1m','>2m','10-5m',],[1,2,3,4,5])
    x9 = df['补给段长度比'].replace(['<10%','30-10%','60-30%','>60%'],[1,2,3,4])
    x10 = df['松散物平均厚'].replace(['<1m','5-1m','10-5m','>10m'],[1,2,3,4])
    x11 = df['沟槽横断面'].replace(['平坦型','复式断面','拓宽U型谷','V型谷,谷中谷,U型谷'],[1,2,3,4])
    x12 = df['主沟纵坡'].replace(['<3°(<52‰)','6-3°(105-52‰)','12-6°(213-105‰)','>12°(>213‰)'],[1,2,3,4])
    x13 = df['新构造影响'].replace(['沉降区,构造影响小或无影响','相对稳定区,4级以下地震区,有小断层','抬升区,4-6级地震区,有中小支断层或无断层','强抬升区,6级以上地震区'],[1,2,3,4])
    #x14 =  df['泥石流规模等级'].replace(['小型','中型','大型','巨型'],[1,2,3,4])
    
    df1 = pd.DataFrame(np.random.randn(1778,13), columns=['不良地质现象','植被覆盖率','岩性因素','松散物储量','山坡坡度','流域面积','相对高差','冲淤变幅','补给段长度比'
                                                        ,'松散物平均厚','沟槽横断面','主沟纵坡','新构造影响'])
    
    df3 = pd.DataFrame(np.random.randn(1760,1), columns=['泥石流规模等级'])
    df4 = pd.DataFrame(np.random.randn(1778,1), columns=['泥石流易发程度'])
    df1.iloc[:,0] = x1
    df1.iloc[:,1] = x2
    df1.iloc[:,2] = x3
    df1.iloc[:,3] = x4
    df1.iloc[:,4] = x5
    df1.iloc[:,5] = x6
    df1.iloc[:,6] = x7
    df1.iloc[:,7] = x8
    df1.iloc[:,8] = x9
    df1.iloc[:,9] = x10
    df1.iloc[:,10] = x11
    df1.iloc[:,11] = x12
    df1.iloc[:,12] = x13
    df4.iloc[:,0] = y
    #df3.iloc[:,0] = x14
    df4.to_excel(r"E:\山地\泥石流规模1\label131.xlsx",index = False)
    #df3.to_excel(r"E:\山地\泥石流规模\label12.xlsx",index = False)
    df1.to_excel(r"E:\山地\泥石流规模1\整理泥石流132.xlsx",index = False)
    return df1,df4


# In[22]:

df1,df4 =creation_data()


# In[9]:

df3


# In[22]:

individual=[1,2,3,4,5,6,7,8,9]


# In[23]:

b1 = individual[2:7]


# In[24]:

b1


# In[ ]:



