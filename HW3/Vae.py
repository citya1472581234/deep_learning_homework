# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import utils
import matplotlib.pyplot as plt
import time
#%%
HEIGHT, WIDTH, CHANNEL = 64,64,3
BATCH_SIZE = 64
EPOCH = 20
channels = 3
output = HEIGHT*WIDTH*CHANNEL
n_hidden1 = 600 * channels
n_hidden2 = 600 * channels
n_hidden3 = 30 * channels  # codings
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
version='animation_vae'
version_re='animation_vae_re'
animation_path = './' + version
animation_path_re = './' + version_re
total_loss=[]
#%%

def FC_layer(layer_name, x,out_nodes,relu=True):
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value
    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                             initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size])
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        if relu:
            x = tf.nn.elu(x)
        return x
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

    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image = np.multiply(image, 1.0 / 255.0)
    image = tf.reshape(image,[HEIGHT*WIDTH*CHANNEL])

  
    # batch 
    iamges_batch = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    num_images = len(images)
    return iamges_batch, num_images

#%%
def Encode(x,reuse=False):
    with tf.variable_scope('Decode') as scope:
        if reuse:
            scope.reuse_variables()
        fc1 = FC_layer('fc1',x,n_hidden1)
        bn1 = tf.contrib.layers.batch_norm(fc1,
                                           epsilon=1e-5, decay = 0.9,
                                           updates_collections=None, 
                                           scope='bn1')
        fc2 = FC_layer('fc2',bn1,n_hidden2) 
        bn2 = tf.contrib.layers.batch_norm(fc2,
                                           epsilon=1e-5, decay = 0.9,
                                           updates_collections=None, 
                                           scope='bn2')
        mean = FC_layer('mean', bn2, n_hidden3, relu=False)
        gamma = FC_layer('gamma', bn2, n_hidden3, relu=False)
        return mean, gamma
def Decode(x,reuse=False):
     with tf.variable_scope('Encode') as scope:
         if reuse:
             scope.reuse_variables()
         
         fc3 = FC_layer('fc3',x,n_hidden4)
         bn3 = tf.contrib.layers.batch_norm(fc3,
                                           epsilon=1e-5, decay = 0.9,
                                           updates_collections=None, 
                                           scope='bn3')
         fc4 = FC_layer('fc4',bn3,n_hidden5)
         bn4 = tf.contrib.layers.batch_norm(fc4,
                                           epsilon=1e-5, decay = 0.9,
                                           updates_collections=None, 
                                           scope='bn4')
         logits = FC_layer('logits', bn4,output,relu=False)
         pred = tf.sigmoid(logits)
         image = tf.reshape(pred,[-1,HEIGHT,WIDTH,CHANNEL])
         return pred,image


def train():
    with tf.variable_scope('input'):
        #real and fake image placholders
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT*WIDTH*CHANNEL], name='real_image')
    z_mean,z_sigma = Encode(real_image)
    shape=tf.shape(z_sigma)
    
    noise = tf.random_normal(shape,mean=0, stddev=1, dtype=tf.float32)
    x = z_mean +  tf.exp(0.5 * z_sigma) * noise
    logits,x_reconstructed = Decode(x)

    epsilon = 1e-10
    recon_loss = -tf.reduce_sum(real_image * tf.log(epsilon+logits) + \
                                (1-real_image) * tf.log(epsilon+1-logits), axis=1)
    recon_loss = tf.reduce_mean(recon_loss)
    latent_loss = -0.5 * tf.reduce_sum(1 + z_sigma - tf.square(z_mean) - tf.exp(z_sigma), axis=1)
    latent_loss = tf.reduce_mean(latent_loss)
    loss = tf.reduce_mean(recon_loss + latent_loss)


    trainer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'Decode' in var.name]
    g_vars = [var for var in t_vars if 'Encode' in var.name]
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
            train_image = sess.run(image_batch)
            # Update the discriminator
            sess.run(d_clip)
            _, Loss = sess.run([trainer, loss],feed_dict={real_image: train_image})
            total_loss.append(Loss)
            if j %2 == 0:
                print('epoch:%d,train:[%d],loss:%f' % (i,j,Loss))        
        if i%5 == 0:
            if not os.path.exists('./model/' + version):
                os.makedirs('./model/' + version)
            saver.save(sess, './model/' +version + '/' + str(i))  
        if i%1 == 0:
            # save images
            if not os.path.exists(animation_path):
                os.makedirs(animation_path)
            if not os.path.exists(animation_path_re):
                os.makedirs(animation_path_re)
            codings_rnd = np.random.normal(size=[BATCH_SIZE, n_hidden3])
            imgsample = sess.run(x_reconstructed, feed_dict={x: codings_rnd})
            img_reconstructed = sess.run(x_reconstructed,feed_dict={real_image: train_image})
            utils.save_images(imgsample, [8,8] ,animation_path + '/epoch' + str(i) + '.jpg')
            utils.save_images(img_reconstructed, [8,8] ,animation_path_re + '/epoch' + str(i) + '.jpg')
        end=time.time()
        print(' sec/per epoch:',(end-start))
    coord.request_stop()
    coord.join(threads)
    plt.grid(True)
    plt.plot(total_loss,label='learning curve')
    plt.legend(loc='upper right')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()
    
    
if __name__ == "__main__":
    train()