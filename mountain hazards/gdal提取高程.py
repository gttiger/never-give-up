
# coding: utf-8

# In[13]:

from osgeo import gdal


# In[14]:

filename=r'E:\山地\dem\ArcGIS\dem数据\085\ASTGTM2_N28E085_dem.tif'
dataset=gdal.Open(filename)  


# In[15]:

cols=dataset.RasterXSize


# In[18]:

rows=dataset.RasterYSize


# In[20]:

geotransform=dataset.GetGeoTransform()


# In[22]:

bands=dataset.RasterCount


# In[24]:

band=dataset.GetRasterBand(1) 


# In[26]:

data=band.ReadAsArray(0,0,cols,rows) 
print(data)

# In[ ]:



