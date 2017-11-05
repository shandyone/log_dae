#coding=gbk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import read_csv

#—————————————————import data——————————————————————
data = read_csv.data
train_end,test_begin=500,400

#--------------  generate training dataset and test dataset------------
time_step=30      #time step
rnn_unit=100       #hidden layer units
batch_size=60
input_size=1
output_size=1
lr=0.0006         #learning rate
predict_day=7
num_layers=4
epoch=4000


#-----------------------------train data------------------------------
def train_data():
    train_begin=0
    train_x,train_y=[],[]   #training dataset
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)
    for i in range(len(normalized_train_data)-time_step-predict_day):
        if i % batch_size==0:
            batch_index.append(i)
        x1=normalized_train_data[i:i+time_step]
        y1=normalized_train_data[i+predict_day:i+time_step+predict_day]
        train_x.append(x1.tolist())
        train_y.append(y1.tolist())
    batch_index.append((len(normalized_train_data)-time_step-predict_day))
    return train_x,train_y,batch_index


#----------------------------test data--------------------------------
def test_data():
    test_x,test_y=[],[]
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std
    size=(len(normalized_test_data)+time_step-predict_day)//time_step
    for i in range(size-1):
        x2 = normalized_test_data[i * time_step:(i + 1) * time_step]
        y2 = normalized_test_data[i * (time_step+predict_day):(i + 1) * (time_step+predict_day)]
        test_x.append(x2.tolist())
        test_y.extend(y2.tolist())
    test_x.append(normalized_test_data[(i + 1) * time_step:].tolist())
    test_y.extend(normalized_test_data[(i + 1) * (time_step+predict_day):].tolist())
    return test_x,test_y,mean,std


#----------------  define nn variable -------------
X=tf.placeholder(tf.float32, [None,time_step,input_size])    #input
Y=tf.placeholder(tf.float32, [None,time_step,output_size])   #label

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }


#——————————————————difine nn——————————————————
def lstm(batch):      #para
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])

    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    cell=tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=0.5)
    cell=tf.contrib.rnn.MultiRNNCell([cell]*num_layers)#num_layers LSTM layers

    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #as the input of the output layer
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



#——————————————————train model——————————————————
def train_lstm():
    global batch_size
    train_x,train_y,batch_index=train_data()
    with tf.variable_scope("lstm") as scope1:
        pred,final_states=lstm(batch_size)
    #loss function
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        # try:
        #     module_file=tf.train.latest_checkpoint('.')
        #     saver.restore(sess,module_file)
        # except:
        #     sess.run(tf.global_variables_initializer())
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            for step in range(len(batch_index)-2):
                X_train=np.array(train_x[batch_index[step]:batch_index[step+1]])[:,:,np.newaxis]#.reshape([-1,time_step,input_size])
                Y_train=np.array(train_y[batch_index[step]:batch_index[step+1]])[:,:,np.newaxis]#.reshape([-1,time_step,output_size])
                final_states,loss_=sess.run([train_op,loss],feed_dict={X:X_train,Y:Y_train})
                #print (i,loss_)
            print (i,loss_)
            if (i+1) % 20==0:
                #print ("save_model:",saver.save(sess,'./model/stock2.model',global_step=i))
                print "save_model:", saver.save(sess, 'model_lstm/lstm.model', global_step=i)

#——————————————————prediction model——————————————————
def prediction():
    test_x,test_y,mean,std=test_data()
    with tf.variable_scope("lstm", reuse=True) as scope2:
        pred,final_states=lstm(1)      #just input[1,time_step,input_size]
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        module_file=tf.train.latest_checkpoint('model_lstm')
        saver.restore(sess,module_file)
        #the last line is the training dataset。shape=[1,time_step,input_size]
        test_predict=[]

        for step in range(len(test_x)-1):
          input_test_x=np.array(test_x[step]).reshape([-1,time_step,output_size])
          #input_test_x=np.array(test_x[step])
          prob=sess.run(pred,feed_dict={X:input_test_x})
          predict=prob.reshape((-1))
          test_predict.extend(predict)

        print ('test_x:',len(test_x))
        print ('test_y:',len(test_y))
        print ('test_predict:',len(test_predict))

        #test_y=np.array(test_y)*np.std(data_test)+np.mean(data_test)
        #test_predict=np.array(test_predict)*np.std(data_test)+np.mean(data_test)
        test_y=np.array(test_y)*std+mean
        test_predict=np.array(test_predict)*std+mean
        plt.figure()


        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.savefig('pic/lstm.png')
        plt.show()

if __name__ == "__main__":
    train_lstm()
    prediction()


'''
def prediction():
    train_test=tf.constant(1)
    with tf.variable_scope("lstm", reuse=True) as scope2:
        pred,final_states=lstm(1,train_test)      #just input[1,time_step,input_size]
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:

        #base_path='~/PycharmProjects/stock'
        #module_file = tf.train.latest_checkpoint(base_path+'checkpoint')
        #saver.restore(sess, module_file)
        saver.restore(sess,'stock.model')

        #the last line is the training dataset。shape=[1,time_step,input_size]
        prev_seq=train_x[-1]
        predict=[]

        #get 100 results
        for i in range(100):
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            predict.append(next_seq[-1])
            prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))

        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.show()
prediction()
'''