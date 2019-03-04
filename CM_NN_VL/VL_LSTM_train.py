# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:02:22 2018

#!/usr/bin/env python3
@author: maximoskaliakatsos-papakostas
"""

'''
binary columns to decimal
lala = composition.transpose().dot(1 << np.arange(composition.shape[0] - 1, -1, -1))
'''

# from music21 import *
import numpy as np
import score2np as s2n
# import matplotlib.pyplot as plt
import tensorflow as tf
import os

# MAKE DATA ===================================================================
folderName = 'BachChorales'

# the user should give the parts to be considered for multi-hot output
parts_for_surface = [-1, -2]

# time resolution should be set by the user
time_res = 16

all_matrices = s2n.get_concat_VL_np_from_folder(folderName, parts_for_surface, time_res, transpose=True)

# PREPARE DATA FOR LSTM =======================================================
# test batch generation example
max_len = 16
batch_size = 320
step = 1
input_rows = all_matrices.shape[0]
output_rows = 128
num_units = [128, 64]
learning_rate = 0.001
epochs = 5000
temperature = 0.5

all_training_errors = np.zeros(epochs)

# divide data in input-output pairs
input_mats = []
output_mats = []

for i in range(0, all_matrices.shape[1] - max_len, step):
    input_mats.append(all_matrices[:,i:i+max_len])
    output_mats.append(all_matrices[(output_rows-input_rows):,i+max_len])

# make training and testing tensors
train_data = np.zeros((len(input_mats), max_len, input_rows))
target_data = np.zeros((len(output_mats), output_rows))
for i in range(len(input_mats)):
    train_data[i,:,:] = input_mats[i].transpose()
    target_data[i,:] = output_mats[i]

# make batches
num_batches = int( all_matrices.shape[1]/batch_size )
count = 0
all_batches = [] # actually don't need it
for _ in range(num_batches):
    train_batch, target_batch = train_data[count:count+batch_size], target_data[count:count+batch_size]
    count += batch_size
    # sess.run([optimizer] ,feed_dict={x:train_batch, y:target_batch})

# save train batch for getting seed
seed = train_batch[:1:]
np.savez('saved_data/training_data.npz', all_matrices=all_matrices, seed=seed)

tf.reset_default_graph()

# LSTM ========================================================================
def rnn(x, weight, bias, input_rows):
    '''
     define rnn cell and prediction
    '''
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, input_rows])
    x = tf.split(x, max_len, 0)
    
    cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n) for n in num_units]
    stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
    outputs, states = tf.contrib.rnn.static_rnn(stacked_rnn_cell, x, dtype=tf.float32)
    prediction = tf.matmul(outputs[-1], weight) + bias
    return prediction
# end rnn

def sample(predicted_in):
    # keep only positive
    predicted = np.zeros(len(predicted_in))
    passes = np.where(predicted_in >= 0.0)[0]
    next_event = np.zeros( (len(predicted),1) )
    if len(passes) > 4:
        # get the 4 most possible events
        predicted[ passes ] = predicted_in[ passes ]
        passes = predicted.argsort()[-4:][::1]
    elif len(passes) > 0:
        passes = passes[0:np.min([4, len(passes)])]
    
    next_event[passes] = 1
    return next_event
# end sample

x = tf.placeholder("float", [None, max_len, input_rows])
y = tf.placeholder("float", [None, output_rows])
weight = tf.Variable(tf.random_normal([num_units[-1], output_rows]))
bias = tf.Variable(tf.random_normal([output_rows]))

prediction = rnn(x, weight, bias, input_rows)
dist = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)
# dist = tf.reduce_mean(tf.norm(prediction - y, ord=10.0))
# dist = tf.nn.sigmoid_cross_entropy(logits=prediction, multiclass_labels=y)
cost = tf.reduce_mean(dist)
# cost = tf.reduce_mean(tf.squared_difference(prediction, y))
# cost = tf.reduce_mean(tf.norm(prediction - y, ord=10.0))
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

num_batches = int(len(train_data)/batch_size)

min_cost = 10000.0

for i in range(epochs):
    print("----------- Epoch", str(i+1), "/", str(epochs), " -----------")
    count = 0
    for i_batch in range(num_batches):
        # print("batch: ", str(i_batch+1), "/", str(num_batches), "epoch: ", str(i+1), "/", str(epochs))
        train_batch, target_batch = train_data[count:count+batch_size], target_data[count:count+batch_size]
        count += batch_size
        sess.run([optimizer] ,feed_dict={x:train_batch, y:target_batch})
    cost_value = sess.run([cost] ,feed_dict={x:train_data, y:target_data})
    print( "cost_value: ", cost_value[0] )
    all_training_errors[i] = cost_value[0]
    if min_cost > cost_value[0] and i > 100:
        tmpW_1 = sess.run(weight)
        min_cost = cost_value[0]
        print("saving model")
        # save model
        all_vars = tf.global_variables()
        saver = tf.train.Saver()
        saver.save(sess, 'saved_model/file.ckpt')
        '''
        directory = 'all_saved_models/epoch_' + str(i) + '/saved_model/'
        if not os.path.exists(directory):
            os.makedirs(directory)
            saver.save(sess, directory + 'file.ckpt')
        '''

# save training errors
# np.savez('saved_results/training_erros.npz', all_training_errors=all_training_errors)