import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import time
import pickle
import math

from model import TESnet
from config import config
from utils import write_logs, get_data_set, test_parse, train_parse

# CIFAR-10 Parameters
height = config.IMG.height
width = config.IMG.width
num_classes = config.IMG.num_classes

# Directories
logs_sv = config.SV_TRAIN.logs_train
sv_model_dir = config.SV_TRAIN.model_dir
logs_dir = config.SV_TRAIN.logs_dir

# Hyper Parameters
dropout = config.SV_TRAIN.dropout
num_epoches = config.SV_TRAIN.num_epoches
batch_size = config.SV_TRAIN.batch_size
lr_init = config.SV_TRAIN.learning_rate_init
lr_start_decay = config.SV_TRAIN.lr_start_decay
lr_decay = config.SV_TRAIN.lr_decay
lr_decay_period = config.SV_TRAIN.lr_decay_period

def supervised_training():
	train_x, train_y = get_data_set("train")
	with open('..\\data\\svtrain.p', 'rb') as fp:
		idx = pickle.load(fp)
	train_x = train_x[idx, :]
	train_y = train_y[idx, :]
	num_train = train_x.shape[0]

	train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
	train_data = train_data.shuffle(num_train)
	train_data = train_data.map(train_parse, num_parallel_calls=8)
	train_data = train_data.batch(batch_size)
	train_iter = train_data.make_initializable_iterator()
	x_train, y_train = train_iter.get_next()

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

	# Learning Rate
	with tf.variable_scope('learning_rate'):
		lr_v = tf.Variable(lr_init, trainable=False)

	# Loss Function
	with tf.name_scope("Cross_Entropy_Loss"):
		loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	sum_loss_op = tf.summary.scalar("Cross_Entropy_Loss", loss_op)

	# Optimizer
	optimizer = tf.train.AdamOptimizer(lr_v)
	gvs = optimizer.compute_gradients(loss_op)
	capped_gvs = [(tf.clip_by_value(grad,-1.0, 1.0), var) for grad, var in gvs]
	train_op = optimizer.apply_gradients(capped_gvs)

	saver = tf.train.Saver()

	# Evaluate Model
	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	sum_acc_op = tf.summary.scalar("Accuracy", accuracy)

	num_batches = int(math.ceil(num_train/batch_size))

	with tf.Session() as sess:
		log = "\n========== Supervised Training Begin ==========\n"
		write_logs(logs_sv, log, True)
		train_start = time.time()

		# Initialize variables
		sess.run(tf.global_variables_initializer())
		
		# Op to write logs to Tensorboard
		train_sum_writer = tf.summary.FileWriter(logs_dir, tf.get_default_graph())

		for epoch in range(num_epoches):
			epoch_start = time.time()

			if (epoch == 70):
				new_lr = lr_v * lr_decay
				sess.run(tf.assign(lr_v, new_lr))
				log = "** New learning rate: %1.9f **\n" % (lr_v.eval())
				write_logs(logs_sv, log, False)
			elif epoch == 0:
				sess.run(tf.assign(lr_v, lr_init))
				log = "** Initial learning rate: %1.9f **\n" % (lr_init)
				write_logs(logs_sv, log, False)

			avg_loss = 0
			avg_acc = 0

			sess.run(train_iter.initializer)
			for batch in range(num_batches):
				batch_start = time.time()

				bx, by = sess.run([x_train, y_train])
				sess.run([train_op], feed_dict={X:bx, Y:by, drop_rate:dropout})
				loss, acc, sum_loss, sum_acc = sess.run([loss_op, accuracy, sum_loss_op, sum_acc_op], feed_dict={X:bx, Y:by, drop_rate:0.0})

				avg_loss += loss
				avg_acc += acc

				train_sum_writer.add_summary(sum_loss, epoch*num_batches+batch)
				train_sum_writer.add_summary(sum_acc, epoch*num_batches+batch)

				log = "Time {:2.5f}, Epoch {}, Batch {}, Loss = {:2.5f}, Training Accuracy = {:0.4f}".format(time.time()-batch_start, epoch, batch, loss, acc)
				write_logs(logs_sv, log, False)

			log = "\nTime {:2.5f}, Epoch {}, Average Loss = {:2.5f}, Training Average Accuracy = {:0.4f}\n"\
				.format(time.time()-epoch_start, epoch, avg_loss/num_batches, avg_acc/num_batches)
			write_logs(logs_sv, log, False)

		log = "\nSupervised Training Time: {:2.5f}".format(time.time()-train_start)
		write_logs(logs_sv, log, False)
		log = "\n========== Supervised Training End ==========\n"
		write_logs(logs_sv, log, False)

		# Save model
		save_path = saver.save(sess, sv_model_dir)
		log = "Model is saved in file: %s" % save_path
		write_logs(logs_sv, log, False)
		
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

