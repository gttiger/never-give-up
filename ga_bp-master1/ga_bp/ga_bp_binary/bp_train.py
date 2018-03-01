'''
Created on 2017年11月21日

@author: ljs
'''
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
inputdimension = 13
outputdimension = 4
jiedian = 14
jiedian2 = 14
NUM = inputdimension*jiedian+jiedian*jiedian2+jiedian2*outputdimension+jiedian+jiedian2+outputdimension
# 添加层
def add_layer(inputs, Weights, biases, activation_function=None):
    # add one more layer and return the output of this layer
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def bp_train(individual):
    tf.reset_default_graph()  # 重置默认图
    graph = tf.Graph()        # 新建空白图
    with graph.as_default() as g:   # 将新建的图作为默认图
        with tf.Session(graph=g):   # Session  在新建的图中运行
            # 需要运行的代码放这里，每次运行都会使用新的图
            column1 = []
            column2 = []
            # with open('泥石流resample.xlsx',"r") as xlsxfile:
            #     reader = csv.reader(xlsxfile)
            #     for row1 in reader:
            #         row = list(map(float,row1))
            #         data = [row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11]]
            #         column1.append(data)
            #         column2.append([row[12]])
            import pandas as pd
            #df = pd.read_excel(r'E:\山地\code\ga_bp-master\ga_bp\泥石流resample.xlsx')
            #data = [df['不良地质现象'], df['植被覆盖率'], df['岩性因素'], df['不良地质现象'], df['松散物储量'], df['山坡坡度'], df['流域面积'],
                    #df['冲淤变幅'], df['补给段长度比'], df['松散物平均厚'], df['沟槽横断面'], df['主沟纵坡']]
            # column1.append(df['不良地质现象'])
            # column1.append(df['植被覆盖率'])
            # column1.append(df['岩性因素'])
            # column1.append(df['不良地质现象'])
            # column1.append(df['松散物储量'])
            # column1.append(df['山坡坡度'])
            # column1.append(df['流域面积'])
            # column1.append(df['冲淤变幅'])
            # column1.append(df['补给段长度比'])
            # column1.append(df['松散物平均厚'])
            # column1.append(df['沟槽横断面'])
            # column1.append(df['主沟纵坡'])


                    #df['冲淤变幅'], df['补给段长度比'], df['松散物平均厚'], df['沟槽横断面'], df['主沟纵坡'])
            #column2.append([df['易发性程度']])
            #x = np.mat(df.iloc[:,0:12])
            #y = np.mat(df['易发性程度'])
            '''
            df1 = pd.read_excel(r'E:\山地\整理第四把\整理泥石流13resample.xlsx')
            df2 = pd.read_excel(r'E:\山地\整理第四把\四标签13.xlsx')
            df5 =pd.read_excel(r'E:\山地\整理第四把\四标签13测试.xlsx')
            df4 =  pd.read_excel(r'E:\山地\整理第四把\整理泥石流13resample测试.xlsx')
            '''
            '''
            # df1 = pd.read_excel(r'E:\山地\正规第三把\泥石流13resample.xlsx')
            # df2 = pd.read_excel(r'E:\山地\正规第三把\四标签131resample.xlsx')
            # df3 = pd.read_excel(r'E:\山地\正规第三把\单标签13测试.xlsx')
            # df5 =pd.read_excel(r'E:\山地\正规第三把\四标签131测试.xlsx')
            # df4 =  pd.read_excel(r'E:\山地\正规第三把\泥石流13测试.xlsx')
            '''

            df1 = pd.read_excel(r'E:\山地\整理第一把\整理泥石流13resample1.xlsx')
            df2 = pd.read_excel(r'E:\山地\整理第一把\四标签13.xlsx')
            df5 = pd.read_excel(r'E:\山地\整理第一把\四标签13测试.xlsx')
            df4 = pd.read_excel(r'E:\山地\整理第一把\测试.xlsx')
            #x = np.mat(df1)
            #正常归一化及还原，精度0.01
            inputdimension = 13


            outputdimension = 4
            min_max_scaler_x = preprocessing.MinMaxScaler()
            feature_train_scaled = min_max_scaler_x.fit_transform(df1)
            feature_test_scaled = min_max_scaler_x.fit_transform(df4)
            #min_max_scaler_y = preprocessing.MinMaxScaler()
            label0_scaled = df2
            label1_scaled = df5
            #label0_scaled = min_max_scaler_y.fit_transform(df2)
            #label1_scaled = min_max_scaler_y.fit_transform(df5)
            #min_max_scaler_z = preprocessing.MinMaxScaler()
            #label2_scaled = min_max_scaler_z.fit_transform(df3)
            #label3_scaled = min_max_scaler_z.fit_transform(df6)
            def cross_validation(trainset, label):
                # #将数据分成训练集，验证集和测试集（此处无验证集）
                train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = train_test_split(trainset, label,
                                                                                                      train_size=0.75,
                                                                                                      random_state=33)
                return train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder


            xs = tf.placeholder(tf.float32, [None, inputdimension])
            #ys = tf.placeholder(tf.float32, [None, 1])
            ys = tf.placeholder(tf.float32, [None, outputdimension])
            w1 = []
            w2 = []
            w3 = []
            for i in range(inputdimension):
                a = individual[jiedian*i:jiedian*i+jiedian]
                w1.append(a)
            for j in range(jiedian):
                weight =individual[jiedian2*j:jiedian2*j + jiedian2]
                #weight = individual[outputdimension*j:outputdimension*j +outputdimension]
                w2.append(weight)
            for z in range(jiedian2):
                weight1 = individual[outputdimension*z:outputdimension*z +outputdimension]
                w3.append(weight1)
            # b1 = individual[120:130]
            # b2 = individual[130]

            b1 = individual[NUM - outputdimension - jiedian - jiedian2:NUM - outputdimension - jiedian2]
            b2 = individual[NUM - outputdimension - jiedian2:NUM - outputdimension]
            b3 = individual[NUM - outputdimension:NUM]
            Weights_1 = tf.Variable(tf.cast(np.mat(w1), dtype=tf.float32),name='Weights_1')
            biases_1 = tf.Variable(tf.cast(np.mat(b1), dtype=tf.float32),name='biases_1')
            Weights_2 = tf.Variable(tf.cast(np.mat(w2), dtype=tf.float32),name='Weights_2')
            biases_2 = tf.Variable(tf.cast(np.mat(b2), dtype=tf.float32),name='biases_2')
            Weights_3 = tf.Variable(tf.cast(np.mat(w3), dtype=tf.float32), name='Weights_2')
            biases_3 = tf.Variable(tf.cast(np.mat(b3), dtype=tf.float32), name='biases_2')
            # 3.定义神经层：隐藏层和预测层
            # add hidden layer 输入值是 xs，在隐藏层有 10 个神经元   
            l1 = add_layer(xs, Weights_1, biases_1, activation_function=tf.nn.tanh)
            # add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
            #l2 = add_layer(l1, 12, 12, n_layer=2, activation_function=tf.nn.tanh)
            #prediction = add_layer(l1, Weights_2, biases_2, activation_function=None)
            l2 = add_layer(l1, Weights_2, biases_2, activation_function=tf.nn.tanh)
            prediction = add_layer(l2, Weights_3, biases_3, activation_function=tf.nn.softmax)
            
            # 4.定义 loss 表达式
            # the error between prediciton and real data    
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                          reduction_indices=[1]))
            #cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                             #reduction_indices=[1]))
            # 5.选择 optimizer 使 loss 达到最小                   
            # 这一行定义了用什么方式去减少 loss，学习率是 0.1       
            train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess = tf.Session()
            # 上面定义的都没有运算，直到 sess.run 才会开始运算
            sess.run(init)
            # 迭代 1000 次学习，sess.run optimizer
            train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(
                feature_train_scaled,
                label0_scaled)
            for i in range(100):
                sess.run(train_step, feed_dict={xs: train_x_disorder, ys: train_y_disorder})
                error = sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder})
                #error = sess.run(cross_entropy, feed_dict={xs: feature_train_scaled, ys: label0_scaled})
                #accuracy1 = sess.run(accuracy, feed_dict={xs: feature_train_scaled, ys: label0_scaled})
                #accuracy2 = sess.run(accuracy, feed_dict={xs: val_x_disorder, ys: val_y_disorder})

            #print ('训练准确率：')
            #print(accuracy1)
            #error = sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder})
            #error = sess.run(cross_entropy, feed_dict={xs: feature_train_scaled, ys: label0_scaled})
            print(1/error)
            return 1/error

if __name__ == '__main__':
    pass
            