from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np

import csv

import gzip
import os
import math
import time
import operator
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt



# step 1 load dataset
test_final = np.loadtxt(open("test_final_nor.csv","rb"),delimiter=",",skiprows=0)

test_final_label = np.loadtxt(open("test_final_label.csv","rb"),delimiter=",",skiprows=0)

# step 2 dataset preprocess

dataset = test_final
data_label = test_final_label

data_size = 80
num_labels = 8
num_channels = 1

num_images = dataset.shape[0]
num_train = round(num_images*0.7)
num_valid = round(num_images*0.1)
num_test  = round(num_images*0.2)

train_dataset = dataset[0:num_train,:].astype(np.float32)
train_labels  = data_label[0:num_train,:].astype(np.float32)

valid_dataset = dataset[num_train:(num_train+num_valid),:].astype(np.float32)
valid_labels  = data_label[num_train:(num_train+num_valid),:].astype(np.float32)

test_dataset = dataset[(num_train+num_valid):,:].astype(np.float32)
test_labels  = data_label[(num_train+num_valid):,:].astype(np.float32)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

# network part
def MSE(predictions, labels):
	return np.mean((predictions-labels)**2)


def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])


'''Build Net'''
batch_size = 16
hidden_nm1 = 200
hidden_nm2 = 20
graph = tf.Graph()

with graph.as_default():

	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size,data_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,8))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	weights1 = tf.Variable(tf.truncated_normal([data_size,hidden_nm1], stddev=1.0))
	biases1 = tf.Variable(tf.zeros([hidden_nm1]))

        weights2 = tf.Variable(tf.truncated_normal([hidden_nm1,hidden_nm2], stddev=1.0))
	biases2 = tf.Variable(tf.zeros([hidden_nm2]))

	weights3 = tf.Variable(tf.truncated_normal([hidden_nm2,8], stddev=1.0))
	biases3 = tf.Variable(tf.zeros([8]))

	def model(data):


		hidden_in = tf.matmul(data, weights1) + biases1
		hidden_out = tf.nn.relu(hidden_in)
		hidden_in = tf.matmul(hidden_out,weights2) + biases2
		hidden_out = tf.nn.relu(hidden_in)
		o = tf.matmul(hidden_out,weights3) + biases3
		return o

	logits = model(tf_train_dataset)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))

start_time = time.time()
nm_steps = 200000
with tf.Session(graph=graph) as session:
 	tf.initialize_all_variables().run()
 	print('Initialized')
 	for step in range(nm_steps):
 		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
 		batch_data = train_dataset[offset:(offset + batch_size), :]
 		batch_labels = train_labels[offset:(offset + batch_size), :]
 		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
 		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
 		if (step % 5000 == 0):
 			print('*'*40)
 			print('Minibatch loss at step %d: %f' % (step, l))
 			print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
 			print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
 	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
 	end_time = time.time()
	duration = (end_time - start_time)/60
	print("Excution time: %0.2fmin" % duration)
