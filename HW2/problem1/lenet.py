# -*- coding: utf-8 -*-
import tensorflow as tf 
#%%
def lenet(x,keep_prob, n_classes,is_train=True,is_pretrain=True):   
    conv1 = conv('conv1_1',keep_prob, x, 96, kernel_size=[5,5], stride=[1,2,2,1], is_pretrain=is_pretrain)
    x = pool('pool1', conv1, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    conv2 = conv('conv2_1',keep_prob, x, 64, kernel_size=[5,5], stride=[1,2,2,1],is_pretrain=is_pretrain)
    x = pool('pool2', conv2, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    fc3 = FC_layer('fc3',keep_prob, x, out_nodes=384,relu=True)
#    fc3_n = batch_norm(fc3,is_train,False)
    fc4 = FC_layer('fc4',keep_prob,fc3,out_nodes=192,relu=True)
#    fc4_n = batch_norm(fc4,is_train,False)
    x = FC_layer('out',keep_prob,fc4,out_nodes=n_classes,relu=False)
    return conv1,conv2,fc3,x

#%%
def conv(layer_name,keep_prob, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1],is_pretrain=True):
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer()
                            ) 
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))      
#        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(2e-3)(w))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.dropout(x,keep_prob)
        x = tf.nn.relu(x, name='relu')       
        return x
#%%
def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x

#%%
def batch_norm(inputs, is_training,is_conv_out=True,decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])   

        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 0.001)
#%%
def FC_layer(layer_name,keep_prob, x,out_nodes,relu=True):
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value
    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
#        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(2e-3)(w))
        flat_x = tf.reshape(x, [-1, size])
        
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.dropout(x,keep_prob)
        if relu==True:
            x = tf.nn.relu(x)
        return x
#%%
def loss(logits, labels):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels,name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss') 
#        tf.add_to_collection('losses', loss)
#        loss = tf.add_n(tf.get_collection('losses'))
        return loss
    
#%%
def accuracy(logits, labels):
  with tf.name_scope('accuracy'):
      correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
      correct = tf.cast(correct, tf.float32)
      accuracy = tf.reduce_mean(correct)
  return accuracy
#%%
def num_correct_prediction(logits, labels):
  correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  correct = tf.cast(correct, tf.int32)
  n_correct = tf.reduce_sum(correct)
  return n_correct
#%%
def optimize(loss, learning_rate, global_step):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op
 