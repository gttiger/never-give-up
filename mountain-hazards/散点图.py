from numpy import *
import pandas as pd
df = pd.read_excel(r'E:\山地\整理第一把\result0.1.xlsx')
import matplotlib
import matplotlib.pyplot as plt

x=[5,6,7,8,9,10,11,12,13,14]
f1 = plt.figure(1)
#plt.subplot(211)
plt.scatter(x[:],df.iloc[:,1], color = 'm',marker ='x',label='首层节点数5')
plt.scatter(x[:],df.iloc[:,2],color = 'r',marker ='*',label='6')
plt.scatter(x[:],df.iloc[:,3], color = 'b',marker ='.',label='7')
plt.scatter(x[:],df.iloc[:,4], color = 'c',marker ='o',label='8')
plt.scatter(x[:],df.iloc[:,5], color = 'black',marker ='+',label='9')
plt.scatter(x[:],df.iloc[:,6], color = 'y',marker ='v',label='10')
plt.scatter(x[:],df.iloc[:,7], color = 'purple',marker ='^', label='11')
plt.scatter(x[:],df.iloc[:,8], color = 'orange',marker ='<', label='12')
plt.scatter(x[:],df.iloc[:,9], color = 'pink',marker ='>', label='13')
plt.scatter(x[:],df.iloc[:,10], color = 'gold',marker ='s', label='14')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.xlabel('第二层节点数')
plt.ylabel('分类准确率')
#plt.title('不同节点数分类准确率')
#legend = "第一层节点数"
plt.legend(loc = 'left')
# plt.plot(x[:],df.iloc[:,1], color = 'm')
# plt.plot(x[:],df.iloc[:,2],color = 'r', label='1')
# plt.plot(x[:],df.iloc[:,3], color = 'b', label='1')
# plt.plot(x[:],df.iloc[:,4], color = 'c', label='1')
# plt.plot(x[:],df.iloc[:,5], color = 'black', label='1')
# plt.plot(x[:],df.iloc[:,6], color = 'y', label='1')
# plt.plot(x[:],df.iloc[:,7], color = 'purple', label='1')
# plt.plot(x[:],df.iloc[:,8], color = 'orange', label='1')
# plt.plot(x[:],df.iloc[:,9], color = 'pink', label='1')
# plt.plot(x[:],df.iloc[:,10], color = 'gold', label='1')
#plt.ylim(0,1350)
plt.xlim(4,18)
# box = plt.gca().get_position()
# plt.gca().set_position([box.x0, box.y0, box.width , box.height* 0.8])
# plt.gca().legend(loc='lower center', bbox_to_anchor=(0.5, 1.2),ncol=5)


plt.show()