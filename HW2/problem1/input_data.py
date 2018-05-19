# -*- coding: utf-8 -*-

import tensorflow as tf
import os

#%%
def get_files(file_dir):
    image_list=[]
    label_list=[]
    for file in os.listdir(file_dir):
        name=file.split(sep='_')
        image_list.append(file_dir+file)
        label_list.append(name[0])
    label=[int(float(i)) for i in label_list]  
    return image_list,label
#%% 
def get_batch(image,label,image_W,image_H,batch_size,capacity,n_classes,distortion,shuffle=True):
    ser=[]
    image = tf.cast(image,tf.string)
    label = tf.cast(label, tf.int32)
    if shuffle==True:
        input_queue = tf.train.slice_input_producer([image,label],shuffle=True)
    else:
        input_queue = tf.train.slice_input_producer([image,label],shuffle=False)
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels =3)
    image = tf.image.resize_images(image,[image_H,image_W])
    image = tf.image.per_image_standardization(image)
    if distortion:
        image1 = tf.image.flip_left_right(image)
        image2 = tf.image.random_hue(image, max_delta=0.5)
        image3 = tf.image.random_brightness(image,max_delta=0.5)
        image4 = tf.image.random_contrast(image,lower=0.2, upper=1.8)
        image5 = tf.image.random_saturation(image,lower=0.2, upper=1.8)
        ser=[image,image1,image2,image3,image4,image5]
        for image in ser:
            image_batch, label_batch = tf.train.batch([image, label],
                                                      batch_size = batch_size,
                                                      num_threads = 4,
                                                      capacity = capacity)
            label_batch = tf.one_hot(label_batch, depth= n_classes)
            label_batch = tf.cast(label_batch, dtype=tf.int32)
            label_batch = tf.reshape(label_batch, [batch_size, n_classes])
            image_batch = tf.cast(image_batch,tf.float32)
            return  image_batch, label_batch,batch_size
    else:
        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size = batch_size,
                                                  num_threads = 4,
                                                  capacity = capacity)
        label_batch = tf.one_hot(label_batch, depth= n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [batch_size, n_classes])
        image_batch = tf.cast(image_batch,tf.float32)
        return  image_batch, label_batch,batch_size
        
    
    
