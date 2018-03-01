import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
##中文显示
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# df1 = pd.read_csv(r'E:\山地\准确率\对比准确率\run_train,tag_accuracy_accuracy对比.csv')
# df2= pd.read_csv(r'E:\山地\准确率\对比准确率\run_test,tag_accuracy_accuracy对比.csv')
# df1 = pd.read_csv(r'E:\山地\准确率\run_train,tag_accuracy_accuracy.csv')
# df2= pd.read_csv(r'E:\山地\准确率\run_test,tag_accuracy_accuracy.csv')
df1 = pd.read_csv(r'E:\山地\准确率\GA-BP\run_train,tag_accuracy_accuracy遗传.csv')
df2= pd.read_csv(r'E:\山地\准确率\GA-BP\run_test,tag_accuracy_accuracy遗传.csv')
x=df1['Step']
y=df1['Value']
x1=df2['Step']
y1 = df2['Value']
plt.figure()
plt.plot(x, y,label='训练准确率')
plt.plot(x1,y1,label='验证准确率')
plt.xlabel('迭代次数')
plt.ylabel('准确率')
plt.legend(loc = 'lower right')

#plt.title('title')
plt.show()