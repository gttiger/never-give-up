'''
Created on 2017年11月21日

@author: ljs
'''
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
# def add_layer(inputs, Weights, biases, activation_function=None):
#     # add one more layer and return the output of this layer
#     Wx_plus_b = tf.matmul(inputs, Weights) + biases
#     if activation_function is None:
#         outputs = Wx_plus_b
#     else:
#         outputs = activation_function(Wx_plus_b)
#     return outputs
def add_layer(inputs, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    global Wei,Bis
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = Wei[n_layer]
            tf.summary.histogram(layer_name + '/weights', Weights)
    with tf.name_scope('biases'):
        biases = Bis[n_layer]
        tf.summary.histogram(layer_name + '/biases', biases)
    with tf.name_scope('Wx_plus_b'):
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs
def bp_object(individual):
    #column1 = []
    global Wei, Bis
    column2 = []
    #with open('泥石流resample.xlsx',"r") as xlsxfile:
    #with open('泥石流resample.xlsx', "r") as xlsxfile:
        # reader = csv.reader(xlsxfile)
        # for row1 in reader:
        #     row = list(map(float,row1))
        #     data = [row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11]]
        #     column1.append(data)
        #     column2.append([row[12]])
    inputdimension = 13
    outputdimension = 4
    checkpoint_dir = r"E:\tensorflow\save"
    checkpoint_steps = 500
    import pandas as pd
    df = pd.read_excel(r'E:\山地\code\ga_bp-master\ga_bp\泥石流resample.xlsx')
    # data = [df['不良地质现象'], df['植被覆盖率'], df['岩性因素'],df['不良地质现象'], df['松散物储量'], df['山坡坡度'], df['流域面积'], df['冲淤变幅'], df['补给段长度比'], df['松散物平均厚'], df['沟槽横断面'], df['主沟纵坡']]
    # column1.append(data)
    #column2.append(df.iloc[:,12])
    #x = np.mat(column1)
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
    df9 = pd.DataFrame(np.random.randn(955, 1),
                       columns=["易发性"])
    #column2.append(df2)
    #y = np.mat(column2)
    #正常归一化及还原，精度0.01
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

    # x_data = scaler_x.fit_transform(df1)
    # scaler_y = preprocessing.MinMaxScaler()
    # y_data = scaler_y.fit_transform(df2)
    # min_max_scaler_z = preprocessing.MinMaxScaler()
    # label2_scaled = min_max_scaler_z.fit_transform(df3)
    # label3_scaled = min_max_scaler_z.fit_transform(df6)
    xs = tf.placeholder(tf.float32, [None, inputdimension])
    ys = tf.placeholder(tf.float32, [None, outputdimension])
    keep_prob = tf.placeholder(tf.float32)
    w1 = []
    w2 =[]
    w3 = []
    for i in range(inputdimension):
        a = individual[jiedian * i:jiedian * i + jiedian]
        w1.append(a)
    for j in range(jiedian):
        weight = individual[jiedian2 * j:jiedian2 * j + jiedian2]
        # weight = individual[outputdimension*j:outputdimension*j +outputdimension]
        w2.append(weight)
    for z in range(jiedian2):
        weight1 = individual[outputdimension * z:outputdimension * z + outputdimension]
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
    Wei = [Weights_1,Weights_2,Weights_3]
    Bis = [biases_1,biases_2,biases_3]

    # 3.定义神经层：隐藏层和预测层
    # add hidden layer 输入值是 xs，在隐藏层有 10 个神经元   
    l1 = add_layer(xs,n_layer=0, activation_function=tf.nn.tanh)
    l2 = add_layer(l1,n_layer=1, activation_function=tf.nn.tanh)
    # add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
    #prediction = add_layer(l1, Weights_2, biases_2, activation_function=None)
    prediction = add_layer(l2,n_layer=2 , activation_function=tf.nn.softmax)
    real_prediction = tf.argmax(prediction, 1)
    # 4.定义 loss 表达式
    # the error between prediciton and real data    
    with tf.name_scope('cross_entropy'):
        # cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
        # reduction_indices=[1]))# loss

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))
        tf.summary.scalar('cross_entropy', cross_entropy)
    #cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     #reduction_indices=[1]))
    # 5.选择 optimizer 使 loss 达到最小                   
    # 这一行定义了用什么方式去减少 loss，学习率是 0.1       
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    saver = tf.train.Saver()
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(r'E:\tensorflow\logs3', sess.graph)
    # 上面定义的都没有运算，直到 sess.run 才会开始运算
    sess.run(init)
    # 迭代 1000 次学习，sess.run optimizer
    isTrain = True
    #isTrain = False
    if isTrain:
        import time
        start_time = time.time()
        for i in range(10000):
            train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
                label0_scaled)
            sess.run(train_step, feed_dict={xs: train_x_disorder, ys: train_y_disorder})
            error = sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder})
            accuracy1 = sess.run(accuracy,feed_dict={xs: train_x_disorder, ys: train_y_disorder})
            accuracy2 = sess.run(accuracy, feed_dict={xs: val_x_disorder, ys: val_y_disorder})
            accuracy3 = sess.run(accuracy, feed_dict={xs: feature_test_scaled, ys: label1_scaled})
            result = sess.run(merged,
                              feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
            writer.add_summary(result, i)
            if (i + 1) % checkpoint_steps == 0:
                print ('训练准确率：')
                print (accuracy1)
            # result1 = sess.run(merged,
            #                    feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
            # writer.add_summary(result1, i)
            result = sess.run(merged,
                              feed_dict={xs: val_x_disorder, ys: val_y_disorder, keep_prob: 1})
            writer.add_summary(result, i)

            if accuracy1 >= 0.96 and accuracy2 >= 0.94 :
                prediction4 = sess.run(real_prediction, feed_dict={xs: feature_test_scaled, ys: label1_scaled, keep_prob: 1})
                accuracy3 = sess.run(accuracy, feed_dict={xs: feature_test_scaled, ys: label1_scaled})
                cost3 = sess.run(cross_entropy, feed_dict={xs: feature_test_scaled, ys: label1_scaled})
                print ('验证准确率：')
                print(accuracy2)
                df9.iloc[:,0] = prediction4
                df9.to_excel(r'E:\山地\整理第一把\预测值1.xlsx')
                print("测试准确率:")
                print(accuracy3)
                print ('测试损失:')
                print(cost3)
                #print('预测值')
                #print(prediction4)
                print(error)
                cost_time = time.time() - start_time
                print ('花掉时间:')
                print(cost_time)
                filepath = r"E:\tensorflow\save\%s_%s_model_%s.ckpt" % (str(inputdimension), str(outputdimension), str(i))
                print(filepath)
                save_path = saver.save(sess, filepath , global_step=i + 1)
                print("Model saved in file: %s" % save_path)
                col = 13
                df3 = pd.DataFrame(np.random.randn(2865, col),
                                   columns=["不良地质现象", "植被覆盖率", "岩性因素", "松散物储量", "山坡坡度", "流域面积",
                                            "相对高差", "冲淤变幅", "补给段长度比", "松散物平均厚", "沟槽横断面", "主沟纵坡",
                                            '新构造影响'])
                df6 = pd.DataFrame(np.random.randn(2865, col),
                                   columns=["不良地质现象", "植被覆盖率", "岩性因素", "松散物储量", "山坡坡度", "流域面积",
                                            "相对高差", "冲淤变幅", "补给段长度比", "松散物平均厚", "沟槽横断面", "主沟纵坡",
                                            '新构造影响'])
                # train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
                # label0_scaled)
                for i in range(col):
                    train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(
                        feature_train_scaled,
                        label0_scaled)
                    train_x_disorder[:, i] = train_x_disorder[:, i] * 1.1
                    # prediction4 = sess.run(prediction, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
                    real_prediction3 = sess.run(real_prediction,
                                                feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
                    df3.iloc[:, i] = real_prediction3
                    cost1 = sess.run(cross_entropy,
                                     feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
                    print(cost1)
                    print("increase:")
                    print(i)
                    # train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
                    # label0_scaled)
                for j in range(col):
                    train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(
                        feature_train_scaled,
                        label0_scaled)
                    train_x_disorder[:,j] = train_x_disorder[:, j] * 0.9
                    # prediction2 = sess.run(prediction, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
                    real_prediction4 = sess.run(real_prediction,
                                                feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
                    df6.iloc[:, j] = real_prediction4
                    cost2 = sess.run(cross_entropy,
                                     feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
                    print(cost2)
                    print("decrease:")
                    print(j)
                    # print(prediction1)
                # print(df3)
                # print(df6)

                # print(prediction1 )
                # print(sess.run(Weights))
                # print (sess.run(accuracy,feed_dict={xs: feature_test_scaled, ys: label1_scaled, keep_prob: 1}))
                # print(sess.run(biases))
                print(sess.run(cross_entropy, feed_dict={xs: feature_test_scaled, ys: label1_scaled, keep_prob: 1}))
                df7 = pd.DataFrame(np.random.randn(2869, col),
                                   columns=["不良地质现象", "植被覆盖率", "岩性因素", "松散物储量", "山坡坡度", "流域面积",
                                            "相对高差", "冲淤变幅", "补给段长度比", "松散物平均厚", "沟槽横断面", "主沟纵坡",
                                            '新构造影响'])
                for a in range(col):
                    df7.iloc[:, a] = df3.iloc[:, a] - df6.iloc[:, a]
                df8 = pd.DataFrame(np.random.randn(1 ,col),
                                   columns=["不良地质现象", "植被覆盖率", "岩性因素", "松散物储量", "山坡坡度", "流域面积",
                                            "相对高差", "冲淤变幅", "补给段长度比", "松散物平均厚", "沟槽横断面", "主沟纵坡",
                                            '新构造影响'])
                first = df7["不良地质现象"].mean()
                second = df7["植被覆盖率"].mean()
                thrid = df7["岩性因素"].mean()
                forth = df7["松散物储量"].mean()
                fifth = df7["山坡坡度"].mean()
                sixth = df7["流域面积"].mean()
                seventh = df7["冲淤变幅"].mean()
                eighth = df7["相对高差"].mean()
                ninth = df7["补给段长度比"].mean()
                tenth = df7["松散物平均厚"].mean()
                eleventh = df7["沟槽横断面"].mean()
                twelfth = df7["主沟纵坡"].mean()
                thirteenth = df7["新构造影响"].mean()
                print ("不良地质现象:")
                print(first)
                print ("植被覆盖率:")
                print (second)
                print("岩性因素:")
                print(thrid)
                print("松散物储量:")
                print (forth)
                print ("山坡坡度:")
                print(fifth)
                print ("流域面积")
                print (sixth)
                print ("冲淤变幅")
                print(seventh)
                print ("相对高差:")
                print(eighth)
                print("补给段长度比")
                print (ninth)
                print("松散物平均厚")
                print (tenth)
                print("沟槽横断面")
                print(eleventh)
                print ("主沟纵坡")
                print (twelfth)
                print  ("新构造影响")
                print(thirteenth)
                zhibiao = [first,second,thrid ,forth ,fifth,sixth,seventh,eighth,ninth,tenth,eleventh,twelfth,thirteenth]
                #print (len(df8))
                for zzz in range(13):
                    df8.iloc[0,zzz] = zhibiao[zzz]
                print(zhibiao)
                print (df8)
                #df8.sort()
                print(df8)
                #df8.to_excel(r'E:\山地\整理第一把\miv.xlsx')
                break

    # else:
    #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #     print(ckpt)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #
    #         print(ckpt)
    #         # prediction = sess.run(prediction, feed_dict={xs: feature_test_scaled, ys: label1_scaled})
    #         # cost = sess.run(loss, feed_dict={xs: train_x_disorder, ys: train_y_disorder})
    #     else:
    #         pass
    #     col = 13
    #     df3 = pd.DataFrame(np.random.randn(2869, col), columns=["不良地质现象", "植被覆盖率", "岩性因素", "松散物储量", "山坡坡度", "流域面积",
    #                                                             "相对高差", "冲淤变幅", "补给段长度比", "松散物平均厚", "沟槽横断面", "主沟纵坡",
    #                                                             '新构造影响'])
    #     df6 = pd.DataFrame(np.random.randn(2869, col), columns=["不良地质现象", "植被覆盖率", "岩性因素", "松散物储量", "山坡坡度", "流域面积",
    #                                                             "相对高差", "冲淤变幅", "补给段长度比", "松散物平均厚", "沟槽横断面", "主沟纵坡",
    #                                                             '新构造影响'])
    #     # train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
    #     # label0_scaled)
    #     for i in range(col):
    #         train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
    #                                                                                               label0_scaled)
    #         train_x_disorder[:, i] = train_x_disorder[:, i] * 1.1
    #         # prediction4 = sess.run(prediction, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    #         real_prediction3 = sess.run(real_prediction,
    #                                     feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    #         df3.iloc[:, i] = real_prediction3
    #         cost1 = sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    #         print(cost1)
    #         print("increase:")
    #         print(i)
    #         # train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
    #         # label0_scaled)
    #     for j in range(col):
    #         train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
    #                                                                                               label0_scaled)
    #         train_x_disorder[:, i] = train_x_disorder[:, i] * 0.9
    #         # prediction2 = sess.run(prediction, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    #         real_prediction4 = sess.run(real_prediction,
    #                                     feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    #         df6.iloc[:, j] = real_prediction4
    #         cost2 = sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    #         print(cost2)
    #         print("decrease:")
    #         print(j)
    #         # print(prediction1)
    #     # print(df3)
    #     # print(df6)
    #
    #     # print(prediction1 )
    #     # print(sess.run(Weights))
    #     # print (sess.run(accuracy,feed_dict={xs: feature_test_scaled, ys: label1_scaled, keep_prob: 1}))
    #     # print(sess.run(biases))
    #     print(sess.run(cross_entropy, feed_dict={xs: feature_test_scaled, ys: label1_scaled, keep_prob: 1}))
    #     df7 = pd.DataFrame(np.random.randn(2869, col), columns=["不良地质现象", "植被覆盖率", "岩性因素", "松散物储量", "山坡坡度", "流域面积",
    #                                                             "相对高差", "冲淤变幅", "补给段长度比", "松散物平均厚", "沟槽横断面", "主沟纵坡",
    #                                                             '新构造影响'])
    #     for a in range(col):
    #         df7.iloc[:, a] = df3.iloc[:, a] - df6.iloc[:, a]
    #     # print(df7)
    #     first = df7["不良地质现象"].mean()
    #     second = df7["植被覆盖率"].mean()
    #     thrid = df7["岩性因素"].mean()
    #     forth = df7["松散物储量"].mean()
    #     fifth = df7["山坡坡度"].mean()
    #     sixth = df7["流域面积"].mean()
    #     seventh = df7["冲淤变幅"].mean()
    #     eighth = df7["相对高差"].mean()
    #     ninth = df7["补给段长度比"].mean()
    #     tenth = df7["松散物平均厚"].mean()
    #     eleventh = df7["沟槽横断面"].mean()
    #     twelfth = df7["主沟纵坡"].mean()
    #     thirteenth = df7["新构造影响"].mean()
    #     print ("不良地质现象:")
    #     print(first)
    #     print ("植被覆盖率:")
    #     print (second)
    #     print("岩性因素:")
    #     print(thrid)
    #     print("松散物储量:")
    #     print (forth)
    #     print ("山坡坡度:")
    #     print(fifth)
    #     print ("流域面积")
    #     print (sixth)
    #     print ("冲淤变幅")
    #     print(seventh)
    #     print ("相对高差:")
    #     print(eighth)
    #     print("补给段长度比")
    #     print (ninth)
    #     print("松散物平均厚")
    #     print (tenth)
    #     print("沟槽横断面")
    #     print(eleventh)
    #     print ("主沟纵坡")
    #     print (twelfth)
    #     print  ("新构造影响")
    #     print(thirteenth)
    # #保存模型
    # saver = tf.train.Saver()
    # model_path = "tf_model/bp_model.ckpt"
    # save_path = saver.save(sess,model_path)
    #计算真实误差
    #prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
    # real_pre = scaler_y.inverse_transform(prediction_value)

    # print("y")
    # print(y)
    # print(df2)
    # print("real_pre")
    # print(real_pre)
    # result = df2 - real_pre
    #print('qqq')
    #print(qqq)
    #result = y - real_pre
    #print(result)
    # re = []
    # re_sum = 0
    # for i in range(3926):
    #     re_sum = re_sum + abs(round(float(result.iloc[i,0]),8))
    #     re.append(round(float(result.iloc[i,0]),8))
    # print(re_sum/3927)
if __name__ == '__main__':
    pass
            