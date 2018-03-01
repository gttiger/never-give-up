# -*- coding: utf-8 -*-
'''
Created on 2017年11月21日

@author: ljs
'''
# from ga_bp_binary.ga_encoding import ga_encoding
# from ga_bp_binary.ga_decoding import ga_decoding,ga_decoding_individual
# from ga_bp_binary.ga_calObject import ga_calObject
# from ga_bp_binary.ga_calFitness import ga_calFitness
# from ga_bp_binary.ga_selection import ga_selection
# from ga_bp_binary.ga_crossover import ga_crossover
# from ga_bp_binary.ga_mutation import ga_mutation
# from ga_bp_binary.ga_replace import ga_replace
# from ga_bp_binary.ga_getBest import ga_getBest
# from ga_bp_binary.bp_object import bp_object

import pandas as pd
import numpy as np
import random
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from ga_bp_binary.ga_encoding import ga_encoding
from ga_bp_binary.ga_decoding import ga_decoding,ga_decoding_individual
from ga_bp_binary.ga_calObject import ga_calObject
from ga_bp_binary.ga_calFitness import ga_calFitness
from ga_bp_binary.ga_selection import ga_selection
from ga_bp_binary.ga_crossover import ga_crossover
from ga_bp_binary.ga_mutation import ga_mutation
from ga_bp_binary.ga_replace import ga_replace
from ga_bp_binary.ga_getBest import ga_getBest
from ga_bp_binary.bp_object import bp_object
import time


# df1 = pd.read_excel(r'E:\山地\正规来一把\泥石流resample.xlsx')
# df2 = pd.read_excel(r'E:\山地\正规来一把\单标签resample.xlsx')
# df5 =pd.read_excel(r'E:\山地\正规来一把\单标签测试.xlsx')
# df4 =  pd.read_excel(r'E:\山地\正规来一把\泥石流resample测试.xlsx')

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
df5 =pd.read_excel(r'E:\山地\整理第一把\四标签13测试.xlsx')
df4 =  pd.read_excel(r'E:\山地\整理第一把\测试.xlsx')



inputdimension = 13
outputdimension = 4
jiedian = 14
jiedian2 = 14
POP_SIZE =10#种群个体数量
GEN = 10#遗传迭代代数
CHROM = 8  #染色体二进制位数
NUM = inputdimension*jiedian+jiedian*jiedian2+jiedian2*outputdimension+jiedian+jiedian2+outputdimension#待优化权值与偏重数量
#NUM = 169
#NUM =238
#NUM = 420
PC = 0.6#交叉概率
PM = 0.1#变异概率
result = [[]]#存储最优解及其对应权值偏重
pop = ga_encoding(POP_SIZE,CHROM,NUM)
start_time = time.time()
# for j in range(13):
#     df1 = pd.read_excel(r'E:\山地\排序\%s.xlsx'%j)
inputdimension = 13
outputdimension = 4
for i in range(GEN):
    x = ga_decoding(pop, CHROM, NUM)
    obj = ga_calObject(x)
    best_pop,best_fit = ga_calFitness(pop,obj)
    #如果这一代最优值高于上一代，就用上一代最优值代替这一代最差的
    if len(result) != 1 and best_fit>result[-1][0]:
        ga_replace(result[-1],pop,obj)
    result.append([best_fit,ga_decoding_individual(best_pop, CHROM, NUM),best_pop])
    #python中list,dict是可变对象，参数传递相当于引用传递，所以会改变变量本身，string,tuple,number是不可变对象
    ga_selection(pop,obj)
    ga_crossover(pop,PC)
    ga_mutation(pop,PM)
    print('over..........%s term'%i)
cost_time = time.time() - start_time
print ('花掉时间:')
print(cost_time)
for r in result:
    print(r)
best = ga_getBest(result)
print(best)
bp_object(best)
print(1)