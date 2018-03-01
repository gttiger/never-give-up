
# coding: utf-8

# In[6]:

import numpy as np
import os

file_path = r'E:\typhoon\best\bwp1984.zip_files'
for filename in os.listdir(file_path):
    dataMat = []
    fr = open(os.path.join(file_path,filename))
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #fltLine = list(map(float,curLine))
        #fltLine1 = list(fltLine)
        dataMat.append(curLine)
        print(dataMat)


# In[ ]:



