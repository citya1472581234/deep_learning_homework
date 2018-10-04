# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import utils
import matplotlib.pyplot as plt
import time
#%%
HEIGHT, WIDTH, CHANNEL = 32,32,3
BATCH_SIZE = 64
EPOCH = 100
version='animation'
animation_path = './' + version
total_d_loss=[]
total_g_loss=[]
total_d_loss_mo=[]
total_g_loss_mo=[]
#%%
def lrelu(x, n, leak=0.2): 
    return tf.maximum(x, leak * x, name=n) 
def preprocess_data():   
    current_dir = os.getcwd()
    animation_dir = os.path.join(current_dir, 'faces')
    images = []
    for picture in os.listdir(animation_dir):
        images.append(os.path.join(animation_dir,picture))
    All_images = tf.convert_to_tensor(images, dtype = tf.string)
    images_queue = tf.train.slice_input_producer([All_images])                                    
    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    
    # augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image = tf.reshape(image,[HEIGHT,WIDTH,CHANNEL])
    image = tf.cast(image, tf.float32)
    image = (image / 127.5)-1
  
    # batch 
    iamges_batch = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    num_images = len(images)
    return iamges_batch, num_images
#%%
def generator(input, random_dim, is_train, reuse=False):
    # channel num
    c4, c8, c16, c32= 1024, 512, 256, 128
    s4 =int( HEIGHT/16 )
    output_dim = CHANNEL 
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        # random vector
        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')

        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1,is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
 
        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5,5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2,is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')

        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5,5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3,is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')

        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5,5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4,is_training=is_train, epsilon=1e-5, decay = 0.9, updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')

        conv5 = tf.layers.conv2d_transpose(act4, output_dim, kernel_size=[5,5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        act5 = tf.nn.tanh(conv5, name='act5')
        
        return act5
def discriminator(input, is_train, reuse=False):
    # channel num
    c8, c16, c32, c64 = 128, 256, 512, 1024  
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()
        conv1 = tf.layers.conv2d(input, c8, kernel_size=[5,5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1,is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
        act1 = lrelu(bn1, n='act1')

        conv2 = tf.layers.conv2d(act1, c16, kernel_size=[5,5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')

        conv3 = tf.layers.conv2d(act2, c32, kernel_size=[5,5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3,is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
        
        conv4 = tf.layers.conv2d(act3, c64, kernel_size=[5,5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')
        

        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')       
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        
        pred = tf.nn.sigmoid(logits)
        return pred,logits

def train():
    random_dim = 100
 
    with tf.variable_scope('input'):
        #real and fake image placholders
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')
    
    fake_image = generator(random_input, random_dim, is_train)
    
    real_result_pred,real_result = discriminator(real_image, is_train)
    fake_result_pred,fake_result = discriminator(fake_image, is_train, reuse=True)
    
    
#    d_loss = -tf.reduce_mean(tf.log(real_result) + tf.log(1. - fake_result))
#    g_loss = -tf.reduce_mean(tf.log(fake_result))     
    # DCLAN
#    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#            logits=real_result, labels=tf.ones_like(real_result)))
#    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#            logits=fake_result, labels=tf.zeros_like(fake_result)))
#    d_loss = D_loss_real + D_loss_fake
#    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#            logits=fake_result, labels=tf.ones_like(fake_result)))
    
    # LSGAN
    D_loss_real = tf.losses.mean_squared_error(tf.ones_like(real_result), real_result)
    D_loss_fake = tf.losses.mean_squared_error(tf.zeros_like(fake_result), fake_result)
    d_loss = (D_loss_real + D_loss_fake) / 2.0
    g_loss = tf.losses.mean_squared_error(tf.ones_like(fake_result), fake_result)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]


    trainer_d = tf.train.AdamOptimizer(learning_rate=2e-4/2,beta1=0.5).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.AdamOptimizer(learning_rate=1e-4/2,beta1=0.5).minimize(g_loss, var_list=g_vars)
    
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]
    batch_size = BATCH_SIZE
    image_batch, samples_num = preprocess_data()
    
    batch_num = int(samples_num / batch_size)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    save_path = saver.save(sess, "/tmp/model.ckpt")
    saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('total training sample num:%d' % samples_num)
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
    

    
    for i in range(EPOCH):
        start=time.time()
        for j in range(batch_num):
            print('batch:',j)
            d_iters = 1
            g_iters = 1
            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            for k in range(d_iters):
                train_image = sess.run(image_batch)
                sess.run(d_clip)
                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})
                total_d_loss.append(dLoss)
            # Update the generator
            for k in range(g_iters):
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})
                total_g_loss.append(gLoss)
            if j%50 == 0:
                print('epoch:%d,train:[%d],d_loss:%f,g_loss:%f' % (i,j, dLoss, gLoss))
            if j%100 == 0:
                # save images
                if not os.path.exists(animation_path):
                    os.makedirs(animation_path)
                sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
                imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
                utils.save_images(imgtest, [8,8] ,animation_path + '/epoch' + str(i)+'_'+str(j)+ '.jpg')
        if (i+1)%25 == 0:
            if not os.path.exists('./model/' + version):
                os.makedirs('./model/' + version)
            saver.save(sess, './model/' +version + '/' + str(i))  
        end=time.time()
        print(' sec/epoch :',(end-start))
    coord.request_stop()
    coord.join(threads)

    
    moving_nstep=5
    for i, (j,k) in enumerate(zip(total_d_loss,total_g_loss)):
        if (i+1) % moving_nstep==0:
            total_d_loss_mo.append(np.convolve(total_d_loss[i-moving_nstep+1:i], np.ones((moving_nstep,))/moving_nstep, mode='valid'))
            total_g_loss_mo.append(np.convolve(total_g_loss[i-moving_nstep+1:i], np.ones((moving_nstep,))/moving_nstep, mode='valid'))
    plt.figure(1)
    plt.grid(True)
    plt.title('smoothed')
    A=plt.plot(total_d_loss_mo,label='Discriminator')
    B=plt.plot(total_g_loss_mo,label='Generator')
    plt.legend(handles=[A, B], labels=['Discriminator', 'Generator'],  loc='upper right')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()
    
    
if __name__ == "__main__":
    train()