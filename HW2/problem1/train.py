# -*- coding: utf-8 -*-
import os
import os.path

import time
import tensorflow as tf
import matplotlib.pyplot as plt

import input_data
import lenet

#%%
IMG_W = 64
IMG_H = 64
traing_len=9866
N_CLASSES = 11
learning_rate = 0.00001
Max_epoch=2
IS_PRETRAIN = True
IS_TRAIN = True
CAPACITY=200
BATCH_SIZE = 128
iterations=(traing_len//BATCH_SIZE)*5
#%%   
def train():
 
    train_dir = 'C:/tensorflow/food/training/'
    vali_dir= 'C:/tensorflow/food/validation/'
    train_log_dir = 'C:/tensorflow/food/log/'

    train, train_label = input_data.get_files(train_dir)  
    test_train,test_train_label = input_data.get_files(vali_dir)
    tr_loss=[]
    with tf.Graph().device('/gpu:0'):
        my_global_step = tf.Variable(0, name='global_step', trainable=False)
        train_image_batch,train_label_batch,_=input_data.get_batch(train,train_label,
                                                                   IMG_W,IMG_H,BATCH_SIZE,
                                                                   CAPACITY,N_CLASSES,
                                                                   distortion=True)  
        test_image_batch,test_label_batch,_=input_data.get_batch(test_train,test_train_label,
                                                                 IMG_W,IMG_H,BATCH_SIZE,
                                                                 CAPACITY,N_CLASSES,
                                                                 distortion=False)   
        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
        y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES]) 
        keep_prob = tf.placeholder(tf.float32)
        _,_,_,logits = lenet.lenet(x,keep_prob,N_CLASSES,IS_TRAIN,IS_PRETRAIN)
        loss = lenet.loss(logits, y_)
        accuracy = lenet.accuracy(logits, y_)
        train_op = lenet.optimize(loss, learning_rate, my_global_step)   
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=Max_epoch)      
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for epoch in range(Max_epoch):
                print('epoch:',epoch)
                for iteration in range(iterations):
                    if coord.should_stop():
                        break        
                    if iteration%50==0:  
                        start = time.time()
                        tra_images,tra_labels= sess.run([train_image_batch, train_label_batch])
                        _, tra_loss, tra_acc = sess.run([train_op,loss,accuracy],
                                                         feed_dict={x:tra_images, y_:tra_labels,keep_prob:1.})
                        val_images, val_labels = sess.run([test_image_batch, test_label_batch])
                        val_loss, val_acc = sess.run([loss, accuracy],
                                                     feed_dict={x:val_images,y_:val_labels,keep_prob:1.})
                        end = time.time()
                        print('iteration: {} '.format(iteration),
                              'loss_train: {:.4f} '.format(tra_loss),
                              'accu_train: {:>5.2%} '.format(tra_acc),
                              'loss_test: {:.4f} '.format(val_loss),
                              'accu_valid: {:>5.2%} '.format(val_acc),
                              '{:.4f} sec/batch'.format((end - start)))  
                        tr_loss.append(tra_loss)
                    else:
                        tra_images,tra_labels= sess.run([train_image_batch, train_label_batch])                    
                        sess.run([train_op],feed_dict={x:tra_images, y_:tra_labels,keep_prob:1.})
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop() 
        sess.close()
        coord.join(threads)
        plt.title('learning curve')
        plt.grid(True)
        plt.plot(tr_loss,'b-')
        plt.xlabel('iteration')
        plt.show()
    
   
    
#%%
def test():
    iterations_te=52
    
    test_dir = 'C:/tensorflow/food/evaluation/'
    train_dir = 'C:/tensorflow/food/training/'
    train, train_label = input_data.get_files(train_dir)  
    test_train,test_train_label = input_data.get_files(test_dir)
    
    train_image_batch,train_label_batch,_=input_data.get_batch(train,train_label,
                                                               IMG_W,IMG_H,BATCH_SIZE,
                                                               CAPACITY,N_CLASSES,
                                                               distortion=False)  
    test_image_batch,test_label_batch,_=input_data.get_batch(test_train,test_train_label,
                                                             IMG_W,IMG_H,BATCH_SIZE,
                                                             CAPACITY,N_CLASSES,
                                                             distortion=False)

    total_te_accuracy=[]
    total_tr_accuracy=[]
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES]) 
    keep_prob = tf.placeholder(tf.float32)      
    _,_,_,logits = lenet.lenet(x,keep_prob, N_CLASSES,False,IS_PRETRAIN)
    accuracy = lenet.accuracy(logits, y_)
    tf.Graph().device('/gpu:0')
    init=tf.global_variables_initializer()
    saver = tf.train.Saver()      
    sess = tf.Session()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for i in range(Max_epoch): 
            saver.restore(sess, 'C:/tensorflow/food/log/model.ckpt-'+str(i))
            te_accuracy=[] 
            train_accuracy=[]
            for iteration in range(iterations_te):
                if coord.should_stop():
                    break
                val_images, val_labels = sess.run([test_image_batch, test_label_batch])
                val_acc = sess.run(accuracy,feed_dict={x:val_images,y_:val_labels,keep_prob:1.})
                te_accuracy.append(val_acc)
            tmp=sum(te_accuracy)/iterations_te
            print('epoch'+str(i)+',test acc: {:>5.2%}'.format(tmp))         
            for iteration in range(iterations):
                if coord.should_stop():
                    break
                train_images, train_labels = sess.run([train_image_batch, train_label_batch])
                train_acc = sess.run(accuracy,feed_dict={x:train_images,y_:train_labels,keep_prob:1.})
                train_accuracy.append(train_acc)
            tmp_tr=sum(train_accuracy)/iterations
            print('epoch'+str(i)+',train acc: {:>5.2%}'.format(tmp_tr))
            total_te_accuracy.append(tmp*100)
            total_tr_accuracy.append(tmp_tr*100)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop() 
    sess.close()
    coord.join(threads)
    plt.title('Accuracy')
    plt.grid(True)
    plt.plot(total_tr_accuracy,'b-')
    plt.plot(total_te_accuracy,'r-')
    plt.xlabel('iteration')
    plt.ylabel('%')
    plt.show()
   
        
        