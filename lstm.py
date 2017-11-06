#coding=gbk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import read_csv

#--------------------import data----------------------------
data = read_csv.data
train_end,test_begin=500,400

#--------------  generate training dataset and test dataset------------
time_step=30      #time step
rnn_unit=100       #hidden layer units
batch_size=60
input_size=1
output_size=1
#lr=0.0006         #learning rate;we choose the decay learning rate instead of constant lr
predict_day=1
num_layers=4
epoch=2000
lambda_var=0.001  #l2 regularizer

#----------------------------get L2 regularition weight-------------------------
def get_weight(shape, lambda_var):
    var = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lambda_var)(var))
    return var

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
         # 'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         # 'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         'in': get_weight([input_size, rnn_unit],lambda_var),
         'out': get_weight([rnn_unit, 1],lambda_var)
         }

biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }


#------------------difine nn--------------------------
def lstm(batch):      #para
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])
    input_rnn=tf.matmul(input,w_in)+b_in
    #input_rnn = tf.contrib.layers.batch_norm(input_rnn, center=True, scale=True, is_training=True)# batch normalization
    input_rnn = tf.layers.batch_normalization(input_rnn)
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])

    #cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    #cell=tf.contrib.layers.dropout(cell,keep_prob=0.5)
    cell=tf.contrib.rnn.BasicLSTMCell(rnn_unit)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
    cell=tf.contrib.rnn.MultiRNNCell([cell]*num_layers)#num_layers LSTM layers

    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #as the input of the output layer
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


#---------------------train model------------------
def train_lstm():
    global batch_size

    train_x,train_y,batch_index=train_data()
    with tf.variable_scope("lstm") as scope1:
        pred,final_states=lstm(batch_size)

    #loss function
    sess_loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    tf.add_to_collection("losses",sess_loss)
    loss = tf.add_n(tf.get_collection("losses"))

    #decay_learning rate
    #lr = 0.0006
    global_step = tf.Variable(tf.constant(0))
    init_global_rate = 0.006
    lr = tf.train.exponential_decay(init_global_rate,global_step=global_step,decay_steps=5,decay_rate=0.9,staircase=True)

    update_ops =  tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
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
                writer = tf.summary.FileWriter("logs/", sess.graph)
                print "save_model:", saver.save(sess, 'model_lstm/lstm.model', global_step=global_step)



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
