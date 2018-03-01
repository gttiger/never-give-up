
# coding: utf-8

# In[35]:

import time
import datetime


# In[36]:

today=time.strftime("%m%d")


# In[37]:

print(today)


# In[38]:

cur_date=datetime.datetime(2005,8,1)-datetime.timedelta(days=1)


# In[43]:

time_format=cur_date.strftime('%Y%m%d')


# In[44]:

print(time_format)


# In[ ]:



