# -*- coding: UTF-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# 标准化层
def batch_norm(x, dims, is_training):
    # 形状
    shape = x.get_shape().as_list()[-dims:]
    # 偏置系数，初始值为0
    beta = tf.get_variable(tf.zeros_initializer()(shape=shape),
        dtype=tf.float32, trainable=True, name='beta')
    # 缩放系数，初始值为1
    gamma = tf.get_variable(tf.ones_initializer()(shape=shape),
        dtype=tf.float32, trainable=True, name='gamma')
    # 计算均值和方差
    moment_dims = list(range(len(x.get_shape()) - dims))
    batch_mean, batch_var = tf.nn.moments(x, moment_dims, name='moments')
    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(0.9)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(is_training,
        lambda: ema.apply([batch_mean, batch_var]),
        lambda: tf.no_op())
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training, mean_var_with_update,
        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    # 标准化:x输入，mean均值，var方差，beta=偏移值，gama=缩放系数，1e-3=防止除零
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return x

# 线性变换层: 输入(...,F)，输出(...,units)
def fc_layer(X, units=128, repeat=False, scope="default"):
    if repeat:
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
            W = tf.get_variable(name='kernel_r',initializer=tf.glorot_uniform_initializer()(shape = [X.shape[-1],units]),dtype = tf.float32, trainable = True) #(F, F1)
            b = tf.get_variable(name='bias_r', initializer=tf.glorot_uniform_initializer()(shape = [units,]),dtype = tf.float32, trainable = True) #(F1,)
    else:
        W = tf.Variable(tf.glorot_uniform_initializer()(shape=[X.shape[-1], units]), dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)
        b = tf.Variable(tf.zeros_initializer()(shape=[units]), dtype=tf.float32, trainable=True, name='bias')  # (F1,)
    Y = tf.matmul(X, W) + b #(...,F)*(F,F1)+(F1,)=>(...,F1)
    return Y

# Dropout 层: 训练时，采用dropout，测试不采用
def dropout_layer(x, drop_rate, is_training):
    x = tf.cond(is_training,
        lambda: tf.nn.dropout(x, rate=drop_rate),
        lambda: x)
    return x

# 激活层: 带有norm层
def activation_layer(X, activation='relu', bn=False, dims=None, is_training=True):
    inputs = X
    if activation!=None:
        if bn:
            inputs = batch_norm(inputs, dims, is_training)
        if activation == 'relu':
            inputs = tf.nn.relu(inputs)
        elif activation == 'sigmoid':
            inputs = tf.nn.sigmoid(inputs)
        elif activation == 'tanh':
            inputs = tf.nn.tanh(inputs)
        else:
            raise ValueError
    return inputs

# 多层fc
def multi_fc(X, activations=['relu'], units=[64], repeat=False, Scope='default', drop_rate=None, bn=False, dims=None, is_training=True):
    num_layer = len(units)
    inputs = X
    for i in range(num_layer):
        if drop_rate is not None:
            inputs = dropout_layer(inputs, drop_rate=drop_rate, is_training=is_training)
        if repeat:
            inputs = fc_layer(inputs, units=units[i], repeat=True, scope=Scope+"_fc_"+str(i))
        else:
            inputs = fc_layer(inputs, units=units[i])

        inputs = activation_layer(inputs, activation=activations[i], bn=bn, dims=dims, is_training=dims)
    return inputs

# (M,P,N)
def multi_targets(X, std, mean, F_out):
    OD = inverse_positive(X, std, mean,F_out)
    # In_preds = tf.reduce_sum(OD, axis=-1) #(M,P,N)
    # Out_preds = tf.reduce_sum(OD, axis=-2) #(M,P,N)
    # return [OD, In_preds, Out_preds]
    return OD

# 逆标准化层 加入一个线性变换
def inverse_positive(X, std, mean, units):
    inputs = X * std + mean
    inputs = fc_layer(inputs, units=units)
    inputs = activation_layer(inputs, activation='relu')
    return inputs

############################  GCN
''''
功能：可是输入静态矩阵，也可以输入动态矩阵，进行图卷积
K = None，表示无切比雪夫公式；K = 4 表示4阶切比雪夫公式
L=(N,N),X=(B,N,F),Output=(B,N,F1)
L=(B,T,N,N),X=(B,T,N,F),Output=(B,T,N,F1)
L=(N,N),X=(B,T,N,F),Output=(B,T,N,F1)
'''
def gcn_layer(L, X, K = None):
    y1 = X
    y2 = tf.matmul(L,y1)
    if K:
        x0, x1 = y1, y2
        total = [x0, x1]
        for k in range(3, K + 1):
            x2 = 2 * tf.matmul(L,x1) - x0
            total.append(x2)
            x1, x0 = x2, x1
        total = tf.concat(total, axis=-1)
        y2 = total
    return y2

# 多层GCN
def multi_gcn(L, X, activations=["relu"], units=[128], Ks = [None], repeat=False, drop_rate=None, bn=False, dims=None, is_training=True):
    num_layer = len(units)
    inputs = X
    for i in range(num_layer):
        if drop_rate is not None:
            inputs = dropout_layer(inputs, drop_rate=drop_rate, is_training=is_training)
        inputs = gcn_layer(L, inputs, K=Ks[i])
        if repeat:
            inputs = fc_layer(inputs, units=units[i], repeat=True, scope="gc_"+str(i))
        else:
            inputs = fc_layer(inputs, units=units[i])
        inputs = activation_layer(inputs, activation=activations[i], bn=bn, dims=dims, is_training=dims)
    return inputs

################################## RNN
def choose_RNNs(unit,type='GRU'):
    if type=="GRU":
        cell = tf.nn.rnn_cell.GRUCell(num_units=unit)
    elif type=="LSTM":
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=unit, state_is_tuple=False)
    elif type =="RNN":
        cell = tf.nn.rnn_cell.BasicRNNCell(num_units=unit)
    else:
        print("Wrong type")
        cell = None
    return cell

# RNN 层: 输入：(B,P,F) 输出: (B,F)
def lstm_layer(X, unit,type='GRU'):
    cell  = choose_RNNs(unit=unit, type=type)
    outputs, last_states = tf.nn.dynamic_rnn(cell=cell, inputs=X, dtype=tf.float32)
    output = outputs[:, -1, :] #(B,F)
    return output

def multi_lstm(X, units, type='GRU'):
    cells = [choose_RNNs(unit=unit, type=type) for unit in units]
    stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, last_states = tf.nn.dynamic_rnn(cell=stacked_cells, inputs=X, dtype=tf.float32)
    output = outputs[:, -1, :] #(B,F)
    return output

############################# ConvLSTM
# ConvLSTM核心本质还是和LSTM一样，将上一层的输出作下一层的输入。
# 不同的地方在于加上卷积操作之后，为不仅能够得到时序关系，还能够像卷积层一样提取特征，提取空间特征。为什么卷积能提取空间特征?
# 这样就能够得到时空特征。并且将状态与状态之间的切换也换成了卷积计算。
class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
  def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=False, peephole=False, data_format='channels_last', reuse=None):
    super(ConvLSTMCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._peephole = peephole
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):
    c, h = state

    x = tf.concat([x, h], axis=self._feature_axis)
    n = x.shape[-1].value
    m = 4 * self._filters if self._filters > 1 else 4
    W = tf.get_variable('kernel_{}_{}'.format(n,m), self._kernel + [n, m])
    y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)

    if not self._normalize:
      y += tf.get_variable('bias_{}_{}'.format(n,m), [m], initializer=tf.zeros_initializer())
    j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

    if self._peephole:
      i += tf.get_variable('W_ci_{}_{}'.format(n,m), c.shape[1:]) * c
      f += tf.get_variable('W_cf_{}_{}'.format(n,m), c.shape[1:]) * c

    if self._normalize:
      j = tf.contrib.layers.layer_norm(j)
      i = tf.contrib.layers.layer_norm(i)
      f = tf.contrib.layers.layer_norm(f)

    f = tf.sigmoid(f + self._forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * self._activation(j)

    if self._peephole:
      o += tf.get_variable('W_co_{}_{}'.format(n,m), c.shape[1:]) * c

    if self._normalize:
      o = tf.contrib.layers.layer_norm(o)
      c = tf.contrib.layers.layer_norm(c)

    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
    return h, state
