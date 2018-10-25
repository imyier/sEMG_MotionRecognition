#===============================================================================
#官方库
#===============================================================================
import os
#===============================================================================
#本地库
from utils.utilities import *
#===============================================================================
#第三方库
import numpy
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
#===============================================================================

# 数据读取与预处理
"""
    从数据库中读取数据 train
    表示训练集，test表示训练集
    x表示输入，y label 为标签，即希望得到的输出
"""
"""
    Todo:先搭建结构,后续补充代码
"""
x_train,y_train,list_ch_train = read_data(data_path="./data/", split="train")
X_test, labels_test, list_ch_test = read_data(data_path="./data/", split="test")

#数据库相关数据设置
#时间序列的长度，即sEMG传感器的采样数
seq_len = 0
#通道数，即有几个传感器同时采集
channels_num = 0
#分类种类数，即数据经过分类后有几种可能的情况，比如有抬手、挥手等等若干种动作
classes_num = 0


# 超参数设置
#数据分批处理，表示每一批一次性处理的数据条数
batch_size = 0
#在训练时，对数据进行复用，epochs为每条数据的复用次数
epochs = 0
#误差BP时的学习率，即权重的调整速度
learning_rate = 0.001
#如果使用dropout对神经网络中的神经元进行随机致死，则设置每个神经元的致死概率
keep_prob = 0

lstm_size = 27
lstm_layers =2


#===============================================================================
#使用tensorflow实现CNN-LSTM算法
inputs_tf = tf.placeholder(tf.float32,[None,seq_len,channels_num])
labels_tf = tf.placeholder(tf.float32,[None,classes_num])
learning_rate_tf = tf.placeholder(tf.float32)
keep_prob_tf = tf.placeholder(tf.float32)

#先输入一个卷积层进行特征提取
conv1_output = tf.layers.conv1d(
    inputs = inputs_tf,
    filters = 18,
    kernel_size = 2,
    strides = 1,
    padding = 'same',
    activation = tf.nn.relu # 设置激活函数
)

lstm_input = tf.transpose(conv1_output,[1,0,2])
lstm_input = tf.reshape(lstm_input,[-1,channels_num*2])
lstm_input = tf.layers.dense(lstm_input,lstm_size,activation=None)
lstm_input = tf.split(lstm_input,seq_len,0)

#设置lstm cell
lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
drop = tf.contrib.rnn.DropoutWrapper(lstm_cell,output_keep_prob = keep_prob_tf)
cell = tf.contrib.rnn.MultiRNNCell([drop]*lstm_layers)
initial_state = cell.zero_state(batch_size,tf.float32)

outputs,final_state = tf.contrib.rnn.static_rnn(
    cell,
    lstm_input,
    dtype=tf.float32,
    initial_state = initial_state
)

logits = tf.layers.dense(outputs[-1],classes_num,name='logits')
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels_tf))
train_op = tf.train.AdamOptimizer(learning_rate_tf)
gradients = train_op.compute_gradients(cost)
capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
optimizer = train_op.apply_gradients(capped_gradients)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_tf, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    
    for e in range(epochs):
        # Initialize 
        state = sess.run(initial_state)
        
        # Loop over batches
        for x,y in get_batches(X_tr, y_tr, batch_size):
            
            # Feed dictionary
            feed = {inputs_tf : x, labels_tf : y, keep_prob_tf : 0.5, 
                    initial_state : state, learning_rate_tf : learning_rate}
            
            loss, _ , state, acc = sess.run([cost, optimizer, final_state, accuracy], 
                                             feed_dict = feed)
            train_acc.append(acc)
            train_loss.append(loss)
            
            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))
            
            # Compute validation loss at every 25 iterations
            if (iteration%25 == 0):
                
                # Initiate for validation set
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                
                val_acc_ = []
                val_loss_ = []
                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    # Feed
                    feed = {inputs_tf : x_v, labels_tf : y_v, keep_prob_tf : 1.0, initial_state : val_state}
                    
                    # Loss
                    loss_v, state_v, acc_v = sess.run([cost, final_state, accuracy], feed_dict = feed)
                    
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)
                
                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(numpy.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(numpy.mean(val_acc_)))
                
                # Store
                validation_acc.append(numpy.mean(val_acc_))
                validation_loss.append(numpy.mean(val_loss_))
            
            # Iterate 
            iteration += 1
    
    saver.save(sess,"checkpoints-crnn/har.ckpt")


# In[14]:


# Plot training and test loss
t = numpy.arange(iteration-1)

plt.figure(figsize = (6,6))
plt.plot(t, numpy.array(train_loss), 'r-', t[t % 25 == 0], numpy.array(validation_loss), 'b*')
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# In[15]:


# Plot Accuracies
plt.figure(figsize = (6,6))

plt.plot(t, numpy.array(train_acc), 'r-', t[t % 25 == 0], validation_acc, 'b*')
plt.xlabel("iteration")
plt.ylabel("Accuray")
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


# ## Evaluate on test set

# In[16]:


test_acc = []

with tf.Session() as sess:
    # Restore
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints-crnn'))
    
    for x_t, y_t in get_batches(X_test, y_test, batch_size):
        feed = {inputs_tf: x_t,
                labels_tf: y_t,
                keep_prob_tf: 1}
        
        batch_acc = sess.run(accuracy, feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.6f}".format(numpy.mean(test_acc)))

