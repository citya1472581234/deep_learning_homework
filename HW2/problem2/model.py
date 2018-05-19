# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def pick_top_n(preds, vocab_size, top_n=3):
    p = np.squeeze(preds)
    # 除了機率最高top_n個，都設為0
    p[np.argsort(p)[:-top_n]] = 0
    # normalization
    p = p / np.sum(p)
    # top_n個中隨機挑一個
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

class CharRNN:
    def __init__(self, num_classes, num_seqs=200, num_steps=50,
                 lstm_size=512, num_layers=3, learning_rate=0.001,
                 grad_clip=5, sampling=False, train_keep_prob=0.8, use_embedding=False, embedding_size=128):
        if sampling is True:
            num_seqs, num_steps = 1, 1
        else:
            num_seqs, num_steps = num_seqs, num_steps

        self.num_classes = num_classes
        self.num_seqs = num_seqs
        self.num_steps = num_steps
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size

        tf.reset_default_graph()
        self.build_inputs()
        self.build_lstm()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='inputs')
            self.targets = tf.placeholder(tf.int32, shape=(self.num_seqs, self.num_steps), name='targets')
            # for dropout
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')   

            # 中文要用embedding
            # 英文字母不用embedding
            if self.use_embedding is False:
                self.lstm_inputs = tf.one_hot(self.inputs, self.num_classes)
            else:
                with tf.device("/cpu:0"):
                    embedding = tf.get_variable('embedding', [self.num_classes, self.embedding_size])
                    self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

    def build_lstm(self):
        # 有dropout的lstm cell，lstm_size是neuron數
        def get_a_cell(lstm_size, keep_prob):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        with tf.name_scope('lstm'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.num_layers)]
            )
            self.initial_state = cell.zero_state(self.num_seqs, tf.float32)

            self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.lstm_inputs, initial_state=self.initial_state)

            # 通過lstm_outputs得到機率
            seq_output = tf.concat(self.lstm_outputs, 1)
            x = tf.reshape(seq_output, [-1, self.lstm_size])

            with tf.variable_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal([self.lstm_size, self.num_classes], stddev=0.1))
                softmax_b = tf.Variable(tf.zeros(self.num_classes))

            self.logits = tf.matmul(x, softmax_w) + softmax_b
            self.proba_prediction = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        with tf.name_scope('loss'):
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_reshaped)
            self.loss = tf.reduce_mean(loss)
        with tf.name_scope('accuracy'):
            output_pred_cls = tf.argmax(self.proba_prediction, dimension=-1)
            output_rshp = tf.reshape(output_pred_cls, self.targets.get_shape())
            correct_prediction = tf.equal(tf.cast(output_rshp, tf.int32), self.targets)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def build_optimizer(self):
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars))

    def train(self, batch_generator, max_steps, save_path, save_every_n, log_every_n, batch_generator_v):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            # Train network
            step = 0
            new_state = sess.run(self.initial_state)
            loss_train_list = []
            loss_valid_list = []
            accu_train_list = []
            accu_valid_list = []
            for x, y in batch_generator:
                step += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, accu_train, _ = sess.run([self.loss,
                                                     self.final_state,
                                                     self.accuracy,
                                                     self.optimizer],
                                                    feed_dict=feed)
                end = time.time()
                if step % log_every_n == 0:
                    x, y = batch_generator_v.__next__()
                    feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: 1.,
                        self.initial_state: new_state}
                    batch_loss_valid, accu_valid = sess.run([self.loss, self.accuracy], feed_dict=feed)
                    loss_train_list.append(batch_loss)
                    accu_train_list.append(accu_train*100)
                    loss_valid_list.append(batch_loss_valid)
                    accu_valid_list.append(accu_valid*100)
                # control the print lines
                    print('epoch: {}/{} '.format(int(step/100), int(max_steps/100)),
                          'loss_train: {:.4f} '.format(batch_loss),
                          'accu_train: {:>5.1%} '.format(accu_train),
                          'accu_valid: {:>5.1%} '.format(accu_valid),
                          '{:.4f} sec/batch'.format((end - start)))
                if (step % save_every_n == 0):
                    self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
                if step >= max_steps:
                    break
            self.saver.save(sess, os.path.join(save_path, 'model'), global_step=step)
            # plot
            ax1 = plt.figure(1).gca()
            ax2 = plt.figure(2).gca()
            t = np.arange(1, 21, 1, dtype=np.int16)

            plt.figure(1)
            plt.plot(t, loss_train_list, label="loss of train set")
            plt.plot(t, loss_valid_list, label="loss of validation set")
            plt.legend()
            ax1.grid(color='b', linestyle='dashed')

            plt.figure(2)
            plt.plot(t, accu_train_list, label="accuracy of train set")
            plt.plot(t, accu_valid_list, label="accuracy of validation set")
            plt.legend()
            ax2.grid(color='b', linestyle='dashed')
            plt.show()

    def sample(self, n_samples, prime, vocab_size):
        samples = [c for c in prime]
        sess = self.session
        new_state = sess.run(self.initial_state)
        preds = np.ones((vocab_size, ))  
        # default prime=[]，輸入前置字串到layer中讓state有東西
        for c in prime:
            x = np.zeros((1, 1))
            # 輸入一個字母
            x[0, 0] = c
            feed = {self.inputs: x,
                    # test set不用dropout
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, vocab_size)
        # 把字母加到samples中
        samples.append(c)

        # 生成n_samples個字母
        for i in range(n_samples):
            x = np.zeros((1, 1))
            x[0, 0] = c
            feed = {self.inputs: x,
                    self.keep_prob: 1.,
                    self.initial_state: new_state}
            preds, new_state = sess.run([self.proba_prediction, self.final_state], feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(c)

        return np.array(samples)

    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
