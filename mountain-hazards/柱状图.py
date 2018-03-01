import matplotlib.pyplot as plt
#中文显示
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
##标签
plt.xlabel("易发等级")
plt.ylabel("灾害数目")
#x轴中文
plt.xticks((0,1,2,3),("不易发","低易发",'中易发','高易发'))
#title
#plt.title("雅鲁藏布江泥石流易发性统计图")

#顶端数字
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.03*height, "%s" % float(height),ha='center')
rect = plt.bar(left = (0,1,2,3),height = (112,366,1194,106),width = 0.35,facecolor="#9999ff",align="center")#,yerr=100,xerr=0.4上下左右误差
#图例
plt.legend((rect,),(u"灾害数目",))
#上下偏移
plt.ylim(0,1350)
plt.xlim(-0.5,3.5)
autolabel(rect)
plt.show()