import tensorflow as tf
import numpy as np
import time
import pickle
import math

from config import config
from utils import write_logs, get_data_set, test_parse
from model import TESnet

# CIFAR-10 Parameters
height = config.IMG.height
width = config.IMG.width
num_classes = config.IMG.num_classes

# Directories
logs_usv = config.USV_TEST.logs_test
usv_model_dir = config.USV_TRAIN.model_dir

def unsupervised_testing():
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
	is_labeled = tf.placeholder(tf.bool)

	# Networks
	logits_std = TESnet(X, "Student", drop_rate, reuse=False, getter=None)
	pred_std = tf.nn.softmax(logits_std)

	logits_tc = TESnet(X, "Teacher", drop_rate, reuse=False, getter=None)
	pred_tc = tf.nn.softmax(logits_tc)

	# Evaluate Model
	crt_pred_std = tf.equal(tf.argmax(pred_std,1), tf.argmax(Y,1))
	acc_std = tf.reduce_mean(tf.cast(crt_pred_std, tf.float32))
	sum_acc_std_op = tf.summary.scalar("Student Accuracy", acc_std)

	crt_pred_tc = tf.equal(tf.argmax(pred_tc,1), tf.argmax(Y,1))
	acc_tc = tf.reduce_mean(tf.cast(crt_pred_tc, tf.float32))
	sum_acc_tc_op = tf.summary.scalar("Teacher Accuracy", acc_tc)
	sum_acc_op = tf.summary.merge([sum_acc_std_op, sum_acc_tc_op])

	saver = tf.train.Saver()

	with tf.Session() as sess:
		# Initialize variables
		sess.run(tf.global_variables_initializer())

		# Restore weights of model
		saver.restore(sess, usv_model_dir)

		log = "\n========== Semi-supervised Testing Begin ==========\n"
		write_logs(logs_usv, log, False)
		test_start = time.time()
		avg_acc_std = 0
		avg_acc_tc = 0
		sess.run(test_iter.initializer)
		for i in range(num_test):
			batch_start = time.time()

			bx, by = sess.run([x_test, y_test])
			acc_std_val, acc_tc_val = sess.run([acc_std, acc_tc], feed_dict={X:bx, Y:by, drop_rate:0.0})
			avg_acc_std += acc_std_val
			avg_acc_tc += acc_tc_val

			log = "Time {:2.5f}, Image {:05d}, Student Accuracy = {:0.4f}, Teacher Accuracy = {:0.4f}"\
				.format(time.time()-batch_start, i+1, acc_std_val, acc_tc_val)
			write_logs(logs_usv, log, False)

		log = "\nTesting Student Accuracy = {:0.4f}, Testing Teacher Accuracy = {:0.4f}\n".format(avg_acc_std/num_test, avg_acc_tc/num_test)
		write_logs(logs_usv, log, False)
		log = "\nSemi-supervised Testing Time: {:2.5f}".format(time.time()-test_start)
		write_logs(logs_usv, log, False)

		sess.close()


