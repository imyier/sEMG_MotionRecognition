#===============================================================================
#官方库
#===============================================================================
import os
#===============================================================================
#本地库
import utils.utilities
#===============================================================================
#第三方库
import numpy
import tensorflow as tf
import matplotlib
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
x_train,y_train = read_data()

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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,))
