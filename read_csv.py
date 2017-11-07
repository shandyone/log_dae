#coding=gbk
from __future__ import division
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

#—————————————————import data——————————————————————
# f=open('log-insight-2016.csv')
# f_test=open('log-insight-2017.csv')
# df=pd.read_csv(f)     #read data
# df_test=pd.read_csv(f_test)
# data=np.array(df['num'])
# data_test=np.array(df_test['num'])
# data=data[::-1]
# data_test=data_test[::-1]
#
# normalize_data=(data-np.mean(data))/np.std(data)  #normalization
# normalize_data=normalize_data[:,np.newaxis]       #add axis
# normalize_data_test=(data_test-np.mean(data_test))/np.std(data_test)  #normalization
# normalize_data_test=normalize_data_test[:,np.newaxis]       #add axis

f1=open('log-insight-2016.csv')
f2=open('log-insight-2017.csv')
df1=pd.read_csv(f1)
df2=pd.read_csv(f2)
data1=np.array(df1['num'])
data1=data1[::-1]
# print(len(data1))
# plt.figure()
# plt.plot(range(len(data1)),data1)
# plt.savefig('data1.jpg')
# plt.show()
data2=np.array(df2['num'])
data2=data2[::-1]
data=np.hstack((data1,data2))

#normalize_data = (data-np.mean(data))/np.std(data)
normalize_data = (data-np.min(data))/(np.max(data)-np.min(data))
data_max = np.max(data)
data_min = np.min(data)
print normalize_data
#print normalize_data