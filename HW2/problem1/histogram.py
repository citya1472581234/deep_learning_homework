# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 23:52:31 2018

@author: USER
"""
#%%
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import lenet
import os

#%%
def histogram():
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('C:/tensorflow/food/log/model.ckpt-6.meta')
        saver.restore(sess, tf.train.latest_checkpoint('C:/tensorflow/food/log/'))
        
        w=[]
        all_vars = tf.trainable_variables()
        for i,v in enumerate(all_vars):
            if i%2==0:
                w=sess.run(all_vars[i])
                w=np.reshape(w,(1,-1))
            else:
                b=sess.run(all_vars[i])
                b=np.reshape(b,(1,-1))
                out=np.hstack((w,b))
                out=np.reshape(out,(-1,1))
                a=v.name
                plt.title(a)
                plt.hist(out,60,facecolor='blue')
                plt.show()

#%%
def get_one_image(train): 
    files = os.listdir(train)
 
    n = len(files)
    ind = np.random.randint(0,n) 
    img_dir = os.path.join(train,files[ind]) 
    name=img_dir.split(sep='_')

    tmp=name[0]

    label=tmp.split(sep='/')[-1]

    image = Image.open(img_dir)  
    plt.imshow(image)
    plt.show()
    image = image.resize([64,64])  
    image = np.array(image)
    return image ,label 

#%%
def evaluate_one_image():  
    tf.reset_default_graph()
    train = 'C:/tensorflow/food/training/'
    image_array,label= get_one_image(train) 
    food=['bread','dairy','dessert','egg','fried_food','meat','noodles','rice','seafood','soup','vegetables']
    with tf.Graph().as_default():  
        N_CLASSES = 11  
        
        image = tf.cast(image_array, tf.float32)  
        image_s = tf.image.per_image_standardization(image)
       
        image_r = tf.reshape(image_s, [1,64,64,3])  
        x = tf.placeholder(tf.float32, shape=[1,64,64, 3])  
        keep_prob = tf.placeholder(tf.float32)
        conv1,conv2,fc3,logits = lenet.lenet(image_r,keep_prob, N_CLASSES,is_train=False,is_pretrain=True)
   
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph('C:/tensorflow/food/log/model.ckpt-19.meta')
            saver.restore(sess, tf.train.latest_checkpoint('C:/tensorflow/food/log/'))
            
            image_test=sess.run(image_r)
            
                
            prediction = sess.run(logits, feed_dict={x: image_test,keep_prob:1.})
            print(prediction)
            max_index = np.argmax(prediction)
            print(max_index)
            label=int(label)
            print(label)
            if max_index==0:  
                print('pred:bread label:%s' %(food[label]) ) 
            elif max_index==1:  
                print('pred:dairy label:%s' %(food[label]))
            elif max_index==2:  
                print('pred:dessert label:%s' %(food[label]))
            elif max_index==3:  
                print('pred:egg label:%s' %food[label])
            elif max_index==4:  
                print('pred:fried food label:%s' %food[label])
            elif max_index==5:  
                print('pred:meat label:%s' %food[label])
            elif max_index==6:  
                print('pred:noodles label:%s' %food[label])
            elif max_index==7:  
                print('pred:rice label:%s' %food[label])
            elif max_index==8:  
                print('pred:seafood label:%s' %food[label])
            elif max_index==9:  
                print('pred:soup label:%s' %food[label])
            elif max_index==10:  
                print('pred:vegetable label:%s' %food[label])
                
#%%
def show():
    train = 'C:/tensorflow/food/evaluation/'
    image_array,label= get_one_image(train)  

    with tf.Graph().as_default():  
    
        N_CLASSES = 11  

        image = tf.cast(image_array, tf.float32)  
    
        image_s = tf.image.per_image_standardization(image)
       
        image_r = tf.reshape(image_s, [1,64, 64, 3])  
        keep_prob = tf.placeholder(tf.float32)
        conv1,conv2,_,logits = lenet.lenet(image_r,keep_prob, N_CLASSES,is_train=False,is_pretrain=True)
     
        x = tf.placeholder(tf.float32, shape=[1,64,64, 3])  

        logs_train_dir = 'C:/tensorflow/food/log/'     
        
        saver = tf.train.Saver()  

        with tf.Session() as sess:  
            q=sess.run(image_r)
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)  
            if ckpt and ckpt.model_checkpoint_path:  
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path) 
                print('sucess, training step %s' % global_step)  
            else:  
                print('error') 
                
            feature_map = sess.run(conv2, feed_dict={x: q,keep_prob:1.})
            
            feature_map = tf.reshape(feature_map, [8,8,64])
            images = tf.image.convert_image_dtype (feature_map, dtype=tf.uint8)
            images = sess.run(images)
            plt.figure(figsize=(6, 6))
            for i in np.arange(0, 32):
                plt.subplot(6, 6, i + 1)
                plt.axis('off')
                plt.imshow(images[:,:,i])
            plt.show()
    
    
    