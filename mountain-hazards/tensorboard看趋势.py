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

# df1 = pd.read_excel(r'E:\山地\整理第一把\对比.xlsx')
# df5 =pd.read_excel(r'E:\山地\整理第一把\对比四标签测试.xlsx')

# # df2 = pd.read_excel(r'E:\山地\整理第一把\对比单标签.xlsx')
# # df5 =pd.read_excel(r'E:\山地\整理第一把\对比单测试标签.xlsx')
# df4 =  pd.read_excel(r'E:\山地\整理第一把\对比测试.xlsx')
# df2 = pd.read_excel(r'E:\山地\整理第一把\对比四标签.xlsx')

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

# feature_train_scaled = df1
# feature_test_scaled = df4

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
train_writer = tf.summary.FileWriter(r'E:\tensorflow\不归一' + '/train', sess.graph)
test_writer = tf.summary.FileWriter(r'E:\tensorflow\不归一'+ '/test')

#writer = tf.summary.FileWriter(r'E:\tensorflow\logsqushi', sess.graph)
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()


sess.run(init)
isTrain = True
#isTrain = False
train_steps = 10000
checkpoint_steps = 500
checkpoint_dir = r"E:\tensorflow\save"
start_time = time.time()
for i in range(train_steps):
    # train_x_disorder, val_x_disorder, train_y_disorder, val_y_disorder = cross_validation(feature_train_scaled,
    #                                                                                       label0_scaled)
    if i % 10 == 0:  # 记录测试集的summary与accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict={xs: val_x_disorder, ys: val_y_disorder, keep_prob: 1})
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:  # 记录训练集的summary
        if i % 100 == 99:  # Record execution stats
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step],
                                  feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1},
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            print('Adding run metadata for', i)
        else:  # Record a summary
            summary, _ = sess.run([merged, train_step], feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
            train_writer.add_summary(summary, i)
train_writer.close()
test_writer.close()
accuracy3 = sess.run(accuracy, feed_dict={xs: feature_test_scaled, ys: label1_scaled})
cost3 = sess.run(cross_entropy, feed_dict={xs: feature_test_scaled, ys: label1_scaled})
print("测试准确率:")
print(accuracy3)
print('测试损失:')
cost_time = time.time() - start_time
print('花掉时间:')
print(cost_time)
# sess.run(train_step, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    # #prediction1 = sess.run(prediction, feed_dict={xs: train_x_disorder, ys: train_y_disorder})
    # #real_prediction2 = sess.run(real_prediction, feed_dict={xs: train_x_disorder, ys: train_y_disorder})
    # #cost1 = sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    # #cost2 = sess.run(cross_entropy, feed_dict={xs: val_x_disorder, ys: val_y_disorder, keep_prob: 1})
    # accuracy1 = sess.run(accuracy, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    # accuracy2 = sess.run(accuracy, feed_dict={xs: val_x_disorder, ys: val_y_disorder, keep_prob: 1})
    # result = sess.run(merged,
    #                   feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    # writer.add_summary(result, i)
    # result1= sess.run(merged,
    #                   feed_dict={xs: val_x_disorder, ys: val_y_disorder, keep_prob: 1})
    # writer.add_summary(result1, i)
    #
    # if (i + 1) % checkpoint_steps == 0:
    #     print('训练准确率：')
    #     print(accuracy1)
    # # result1 = sess.run(merged,
    # #                    feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1})
    # # writer.add_summary(result1, i)
    # # result = sess.run(merged,
    # #                   feed_dict={xs: val_x_disorder, ys: val_y_disorder, keep_prob: 1})
    # # writer.add_summary(result, i)
    #
    #
    #     print('验证准确率：')
    #     print(accuracy2)



# accuracy3 = sess.run(accuracy, feed_dict={xs: feature_test_scaled, ys: label1_scaled})
# cost3 = sess.run(cross_entropy, feed_dict={xs: feature_test_scaled, ys: label1_scaled})
# print("测试准确率:")
# print(accuracy3)
# print('测试损失:')
# print(cost3)
# cost_time = time.time() - start_time
# print('花掉时间:')
# print(cost_time)