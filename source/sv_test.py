import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import time
import pickle
import math

from model import TESnet
from config import config
from utils import write_logs, get_data_set, test_parse

# CIFAR-10 Parameters
height = config.IMG.height
width = config.IMG.width
num_classes = config.IMG.num_classes

# Directories
logs_sv = config.SV_TEST.logs_test
sv_model_dir = config.SV_TRAIN.model_dir

def supervised_testing():
	test_x, test_y = get_data_set("test")
	num_test = test_x.shape[0]

	test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
	test_data = test_data.map(test_parse, num_parallel_calls=8)
	test_data = test_data.batch(1)
	test_iter = test_data.make_initializable_iterator()
	x_test, y_test = test_iter.get_next()

	X = tf.placeholder(tf.float32, [None, height, width, 3], name='Input')
	Y = tf.placeholder(tf.int32, [None, num_classes], name='Label')
	drop_rate = tf.placeholder(tf.float32)
	
	logits = TESnet(X, "TESnet", drop_rate, reuse=False)
	pred = tf.nn.softmax(logits)

	saver = tf.train.Saver()

	# Evaluate Model
	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	with tf.Session() as sess:
		# Initialize variables
		sess.run(tf.global_variables_initializer())

		# Restore weights of model
		saver.restore(sess, sv_model_dir)

		log = "\n========== Supervised Testing Begin ==========\n"
		write_logs(logs_sv, log, False)
		test_start = time.time()
		avg_acc = 0
		sess.run(test_iter.initializer)
		for i in range(num_test):
			batch_start = time.time()

			bx, by = sess.run([x_test, y_test])
			acc = sess.run(accuracy, feed_dict={X:bx, Y:by, drop_rate:0.0})
			avg_acc += acc

			log = "Time {:2.5f}, Image {:05d}, Testing Accuracy = {:0.4f}".format(time.time()-batch_start, i+1, acc)
			write_logs(logs_sv, log, False)

		log = "\nTesting Accuracy = {:0.4f}\n".format(avg_acc/num_test)
		write_logs(logs_sv, log, False)
		log = "\nSupervised Testing Time: {:2.5f}".format(time.time()-test_start)
		write_logs(logs_sv, log, False)

		sess.close()

