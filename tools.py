import tensorflow as tf
import numpy as np

def conv(layer_name, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True):
# 每一层卷积结束后，都是用的relu激活函数
# 参数：layer_name：eg. conv1 . pool1
# x: 输入张量，[batch_size, height, width, channels]
# kernel_size 3*3, stride[1,1,1,1]为VGG论文中使用的参数
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights', 
                                   trainable=is_pretrain, 
                                   shape=[kernel_size[0], kernel_size[1], in_channels, out_channels], 
                                   initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='bias', 
                                   trainable=is_pretrain, 
                                   shape=[out_channels], 
                                   initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x

def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
# kernel、stride以及max_pool都是VGG论文中使用的参数，x为输入张量
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x

def batch_norm(x):
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                mean=batch_mean,
                                variance=batch_var,
                                offset=None,
                                scale=None,
                                variance_epsilon=epsilon)
    return x

def FC_layer(layer_name, x, out_nodes):
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value*shape[2].value*shape[3].value
    else:
        size = shape[-1].value
    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size])
        
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        return x

def loss(logits, labels):
# 用来计算loss
# logits：预测值
# labels：标签值，经过one_hot处理
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope+'/loss',loss)
        return loss

def accuracy(logits, labels):
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.arg_max(logits, 1),tf.arg_max(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)*100.0
        tf.summary.scalar(scope+'/accuracy', accuracy)
        return accuracy

def optimize(loss, learning_rate, global_step):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
    
