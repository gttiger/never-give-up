import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd
import time

start_time = time.time()

df1 = pd.read_excel(r'E:\山地\排序\10.xlsx')
df2 = pd.read_excel(r'E:\山地\整理第一把\四标签13.xlsx')
df5 =pd.read_excel(r'E:\山地\整理第一把\四标签13测试.xlsx')
df4 =  pd.read_excel(r'E:\山地\排序1\10.xlsx')


# foo = np.array(df2)
# df2 = foo.flatten()


#from imblearn.over_sampling import SMOTE, ADASYN
#df5 =pd.read_excel(r'E:\山地\正规第三把\四标签131测试.xlsx')
# X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(df1,df2)
# sorted(Counter(y_resampled_smote).items())
# from imblearn.over_sampling import SMOTE, ADASYN
# X_resampled_adasyn, y_resampled_adasyn = ADASYN().fit_sample(df1,df2)
#
# sorted(Counter(y_resampled_adasyn).items())

# from imblearn.over_sampling import SMOTE, ADASYN
# X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(df1,df2)
#
# print(sorted(Counter(y_resampled).items()))

#df4 =  pd.read_excel(r'E:\山地\正规第三把\泥石流13测试.xlsx')
min_max_scaler_x = preprocessing.MinMaxScaler()
min_max_scaler_y = preprocessing.MinMaxScaler()
feature_train_scaled = min_max_scaler_x.fit_transform(df1)
feature_test_scaled = min_max_scaler_x.fit_transform(df4)
label0_scaled = df2

#label2_scaled = min_max_scaler_y.fit_transform(df4)

# df3 = pd.DataFrame(np.random.randn(4776,4),columns = None)
# list1 = [0,0,0,1]
# list2 = [0,0,1,0]
# list3 = [0,1,0,0]
# list4 = [1,0,0,0]
#df_empty = pd.DataFrame()
# for i in range(len(y_resampled_smote)):
#     if y_resampled_smote[i] ==0:
#         df3.iloc[i,:] = list1
#     #df_empty = df_empty.append(df1.iloc[0,0:3])
#     elif y_resampled_smote[i] ==1:
#         df3.iloc[i,:] = list2
#         #df_empty = df_empty.append(df1.iloc[1,0:3])
#     elif y_resampled_smote[i] ==2:
#         df3.iloc[i,:] = list3
#         #df_empty = df_empty.append(df1.iloc[2,0:3])
#     elif y_resampled_smote[i] ==3:
#         df3.iloc[i,:] = list4
#         #df_empty = df_empty.append(df1.iloc[3,0:3])
#     df3.to_excel(r"E:\山地\泥石流易发性\小标签.xlsx",index = False)

#df3 = pd.read_excel(r"E:\山地\泥石流易发性\3.xlsx",index = False)


label1_scaled = df5
def cross_validation(trainset,label):
    # #将数据分成训练集，验证集和测试集（此处无验证集）
    train_x_disorder, val_x_disorder,train_y_disorder , val_y_disorder = train_test_split(trainset, label,
                                                                                            train_size=0.75,random_state=33)
    return train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder
train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
                                                                                               label0_scaled)
#kapa =13
# result = pd.DataFrame(columns=["5", "6", "7", "8", "9", "10","11", "12", "13", "14"])
# #columns=["5", "6", "7", "8", "9", "10",
# result1=[]                                                        #"11", "12", "13", "14"]
# for lu in range(5,15):
#     result1 = []
#     for tao in range(5,15):
inputdimension = 11
jiedian = 11
jiedian2 =12
outputdimension = 4
Weights = tf.Variable(tf.random_normal([inputdimension,jiedian]),name='W')
biases = tf.Variable(tf.zeros([1, outputdimension]) + 0.1,name='b')
def add_layer(inputs, in_size, out_size,n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    global Weights,biases
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name='b')
        tf.summary.histogram(layer_name + '/biases', biases)
    with tf.name_scope('Wx_plus_b'):
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        #Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs



# def compute_accuracy(v_xs, v_ys):
#     global prediction
#     y_pre = sess.run(prediction, feed_dict={xs: v_xs})
#     correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
#     return result
#define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, inputdimension])
    ys = tf.placeholder(tf.float32, [None, outputdimension])
    keep_prob = tf.placeholder(tf.float32)
l1= add_layer(xs,inputdimension ,jiedian, n_layer=1, activation_function=tf.nn.tanh)
l2 = add_layer(l1, jiedian, jiedian2, n_layer=2,activation_function=tf.nn.tanh)
#l3 = add_layer(l2, jiedian, jiedian, n_layer=2,activation_function=tf.nn.tanh)
prediction = add_layer(l2, jiedian2, outputdimension,n_layer=3, activation_function=tf.nn.softmax)
real_prediction = tf.argmax(prediction,1)
letgo =tf.reduce_sum(tf.square(tf.argmax(ys,1)-tf.argmax(prediction,1)))
#prediction = tf.nn.softmax(prediction)
#loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
with tf.name_scope('cross_entropy'):
    #cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        #reduction_indices=[1]))# loss
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                                  reduction_indices=[1]))
    tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
#letgo = tf.equal(tf.argmax(ys,1),tf.argmax(prediction,1))
#Accuracy = tf.reduce_mean(tf.cast(letgo, tf.float32))
#rl_prediction = tf.subtract(tf.argmax(ys,1),tf.argmax(prediction,1))
#real_prediction  = tf.cast(rl_prediction, tf.float32)
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(ys,1),tf.argmax(prediction,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
saver = tf.train.Saver()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(r'E:\tensorflow\logs1', sess.graph)
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()


sess.run(init)
isTrain = True
#isTrain = False
train_steps = 20000
checkpoint_steps = 500
checkpoint_dir = r"E:\tensorflow\save"
start_time = time.time()
for i in range(train_steps):
    # train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
    #                                                                                       label0_scaled)
    batch_xs, batch_ys = train_x_disorder, train_y_disorder
    sess.run(train_step, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    #prediction1 = sess.run(prediction, feed_dict={xs: train_x_disorder, ys: train_y_disorder})
    #real_prediction2 = sess.run(real_prediction, feed_dict={xs: train_x_disorder, ys: train_y_disorder})
    #cost1 = sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    #cost2 = sess.run(cross_entropy, feed_dict={xs: val_x_disorder, ys: val_y_disorder, keep_prob: 1})
    accuracy1 = sess.run(accuracy, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    accuracy2 = sess.run(accuracy, feed_dict={xs: val_x_disorder, ys: val_y_disorder, keep_prob: 1})
    result = sess.run(merged,
                      feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    writer.add_summary(result, i)
    if (i + 1) % checkpoint_steps == 0:
        print('训练准确率：')
        print(accuracy1)
    # result1 = sess.run(merged,
    #                    feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    # writer.add_summary(result1, i)
    # result = sess.run(merged,
    #                   feed_dict={xs: val_x_disorder, ys: val_y_disorder, keep_prob: 1})
    # writer.add_summary(result, i)
    if accuracy1 >= 0.96 and accuracy2 >= 0.94:
        accuracy3 = sess.run(accuracy, feed_dict={xs: feature_test_scaled, ys: label1_scaled})
        cost3 = sess.run(cross_entropy, feed_dict={xs: feature_test_scaled, ys: label1_scaled})
        print('验证准确率：')
        print(accuracy2)

        print("测试准确率:")
        print(accuracy3)
        print('测试损失:')
        print(cost3)
        cost_time = time.time() - start_time
        print('花掉时间:')
        print(cost_time)
        break
    #if (i + 1) % checkpoint_steps == 0:
# #cost1 = sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
# accuracy1 = sess.run(accuracy, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
# #accuracy2 = sess.run(accuracy, feed_dict={xs: val_x_disorder, ys: val_y_disorder , keep_prob: 1})
# accuracy3 = sess.run(accuracy, feed_dict={xs: feature_test_scaled, ys: label1_scaled, keep_prob: 1})

#print ('训练损失:')
#print (cost1)
# print ('验证损失:')
# print (cost2)
#print("训练准确率:%s%s"%(lu,tao))
# print(accuracy1)
# # print ('验证准确率：%s%s'%(lu,tao))
# # print(accuracy2)
# print ('测试准确率：%s%s'%(lu,tao))
#         print (accuracy3)
#         result1.append(accuracy3)
#         print (result1)
#
#     #result =result.append(result1)
#     result[str(lu)]=result1
#     print(result)
# result.to_excel(r'E:\山地\正规第三把\result.xlsx')



            # if accuracy1 >=0.94  and accuracy2>=0.91:
            #     print('预测结果:')
            #     #print(real_prediction2)
            #     cost_time = time.time() - start_time
            #     print ('花掉时间:')
            #     print(cost_time)
            #     accuracy3 = sess.run(accuracy, feed_dict={xs: feature_test_scaled, ys: label1_scaled, keep_prob: 1})
            #     #Accuracy = sess.run(Accuracy, feed_dict={xs: feature_test_scaled, ys: label1_scaled, keep_prob: 1})
            #     cost3 = sess.run(cross_entropy, feed_dict={xs: feature_test_scaled, ys: label1_scaled, keep_prob: 1})
            #     print("测试准确率:")
            #     print(accuracy3)
            #     print ('测试损失:')
            #     print(cost3)
            #     print('shishi；')
            #     print (Accuracy)
            #     filepath = r"E:\tensorflow\save\%s_%s_model_%s.ckpt"%(str(inputdimension) , str(outputdimension) , str(i))
            #     print(filepath)
            #     save_path = saver.save(sess, filepath , global_step=i + 1)
            #     print("Model saved in file: %s" % save_path)
            #     result = sess.run(merged,
            #                       feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
            #     writer.add_summary(result, i)
            #     break
'''
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
print(ckpt)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

    print(ckpt)
    #prediction = sess.run(prediction, feed_dict={xs: feature_test_scaled, ys: label1_scaled})
    #cost = sess.run(loss, feed_dict={xs: train_x_disorder, ys: train_y_disorder})
else:
    pass
col = 11
# #df3 = pd.DataFrame(np.random.randn(2869, col), columns=["不良地质现象", "植被覆盖率", "岩性因素", "松散物储量", "山坡坡度", "流域面积",
#                                                         "相对高差", "冲淤变幅", "补给段长度比", "松散物平均厚", "沟槽横断面", "主沟纵坡",
#                                                         '新构造影响'])
# df3 = pd.DataFrame(np.random.randn(2869, col), columns=["不良地质现象",  "岩性因素", "松散物储量", "山坡坡度", "流域面积",
#                                                          "冲淤变幅", "松散物平均厚", "沟槽横断面",
#                                                         '新构造影响'])
df3 = pd.DataFrame(np.random.randn(2869, col), columns=["不良地质现象", "植被覆盖率", "岩性因素", "松散物储量", "山坡坡度", "流域面积",
                                                         "冲淤变幅", "补给段长度比", "松散物平均厚", "沟槽横断面",
                                                        '新构造影响'])


# df6 = pd.DataFrame(np.random.randn(2869, col), columns=["不良地质现象", "植被覆盖率", "岩性因素", "松散物储量", "山坡坡度", "流域面积",
#                                                         "相对高差", "冲淤变幅", "补给段长度比", "松散物平均厚", "沟槽横断面", "主沟纵坡",
#                                                         '新构造影响'])
df6 = pd.DataFrame(np.random.randn(2869, col), columns=["不良地质现象", "植被覆盖率","岩性因素", "松散物储量", "山坡坡度", "流域面积",
                                                         "冲淤变幅", "补给段长度比", "松散物平均厚", "沟槽横断面",
                                                        '新构造影响'])
# train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
  # label0_scaled)
for i in range(col):
    train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
                                                                                          label0_scaled)
    train_x_disorder[:, i] = train_x_disorder[:, i] * 1.1
    #prediction4 = sess.run(prediction, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    real_prediction3 = sess.run(real_prediction, feed_dict={xs: train_x_disorder, ys: train_y_disorder,keep_prob: 1})
    df3.iloc[:, i] = real_prediction3
    cost1 = sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder,keep_prob: 1})
    print(cost1)
    print("increase:")
    print(i)
    # train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
    # label0_scaled)
for j in range(col):
    train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
                                                                                          label0_scaled)
    train_x_disorder[:, i] = train_x_disorder[:, i] * 0.9
    # prediction2 = sess.run(prediction, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    real_prediction4 = sess.run(real_prediction, feed_dict={xs: train_x_disorder, ys: train_y_disorder,keep_prob: 1})
    df6.iloc[:, j] = real_prediction4
    cost2 = sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder,keep_prob: 1})
    print(cost2)
    print("decrease:")
    print(j)
    # print(prediction1)
#print(df3)
#print(df6)

# print(prediction1 )
# print(sess.run(Weights))
# print (sess.run(accuracy,feed_dict={xs: feature_test_scaled, ys: label1_scaled, keep_prob: 1}))
# print(sess.run(biases))
print(sess.run(cross_entropy, feed_dict={xs: feature_test_scaled, ys: label1_scaled, keep_prob: 1}))

df7 = pd.DataFrame(np.random.randn(2869, col), columns=["不良地质现象","植被覆盖率",  "岩性因素", "松散物储量", "山坡坡度", "流域面积",
                                                         "冲淤变幅", "补给段长度比", "松散物平均厚", "沟槽横断面",
                                                        '新构造影响'])
# df7 = pd.DataFrame(np.random.randn(2869, col), columns=["不良地质现象", "植被覆盖率", "岩性因素", "松散物储量", "山坡坡度", "流域面积",
#                                                         "相对高差", "冲淤变幅", "补给段长度比", "松散物平均厚", "沟槽横断面", "主沟纵坡",
#                                                         '新构造影响'])
for a in range(col):
    df7.iloc[:, a] = df3.iloc[:, a] - df6.iloc[:, a]
#print(df7)
first = df7["不良地质现象"].mean()
second = df7["植被覆盖率"].mean()
thrid = df7["岩性因素"].mean()
forth = df7["松散物储量"].mean()
fifth = df7["山坡坡度"].mean()
sixth = df7["流域面积"].mean()
seventh = df7["冲淤变幅"].mean()
#eighth = df7["相对高差"].mean()
ninth = df7["补给段长度比"].mean()
tenth = df7["松散物平均厚"].mean()
eleventh = df7["沟槽横断面"].mean()
#twelfth = df7["主沟纵坡"].mean()
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
#print ("相对高差:")
#print(eighth)
print("补给段长度比")
print (ninth)
print("松散物平均厚")
print (tenth)
print("沟槽横断面")
print(eleventh)
#print ("主沟纵坡")
#print (twelfth)
print  ("新构造影响")
print(thirteenth)
'''
    # W = sess.run(Weights, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 0.5})

    # cost = sess.run(loss, feed_dict={xs: train_x_disorder, ys: train_y_disorder})
'''
    if (i + 1) % checkpoint_steps == 0:
        save_path = saver.save(sess, r"E:\tensorflow\save\model.ckpt", global_step=i + 1)
        # if i%50 ==0:
        print("Model saved in file: %s" % save_path)
        # cost = sess.run(cross_entropy, feed_dict={xs: val_x_disorder, ys: val_y_disorder, keep_prob: 1})
        cost = sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
        accuracy1 = sess.run(accuracy, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
        # print (prediction1)
        print("accuracy1:")
        print(accuracy1)
        print(cost)
        prediction1 = sess.run(prediction, feed_dict={xs: val_x_disorder, ys: val_y_disorder, keep_prob: 1})
        # min_max_scaler = preprocessing.MinMaxScaler()
        # prediction2 = min_max_scaler_y.inverse_transform(prediction1)
        # prediction2 = inverse(prediction1)
        accuracy1 = sess.run(accuracy, feed_dict={xs: val_x_disorder, ys: val_y_disorder, keep_prob: 1})
        print (prediction1)
        print("accuracy1:")
        print(accuracy1)
        # print(W)
        result = sess.run(merged,
                          feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
        writer.add_summary(result, i)
        cost_time = time.time() - start_time
        print(cost_time)
    '''