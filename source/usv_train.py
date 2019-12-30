import tensorflow as tf
import numpy as np
import time
import pickle
import math

from config import config
from utils import write_logs, get_data_set, test_parse, train_parse
from model import TESnet

# CIFAR-10 Parameters
height = config.IMG.height
width = config.IMG.width
num_classes = config.IMG.num_classes

# Directories
logs_usv = config.USV_TRAIN.logs_train
usv_model_dir = config.USV_TRAIN.model_dir
logs_dir = config.USV_TRAIN.logs_dir

# Hyper Parameters
dropout = config.USV_TRAIN.dropout
labeled_num_epochs = config.USV_TRAIN.labeled_num_epoches
unlabeled_num_epoches = config.USV_TRAIN.unlabeled_num_epoches
batch_size = config.USV_TRAIN.batch_size
lr_init = config.USV_TRAIN.learning_rate_init
lr_start_decay = config.USV_TRAIN.lr_start_decay
lr_decay = config.USV_TRAIN.lr_decay
lr_decay_period = config.USV_TRAIN.lr_decay_period

def unsupervised_training():
	train_x, train_y = get_data_set("train")
	with open('..\\data\\svtrain.p', 'rb') as fp:
		idx = pickle.load(fp)
		fp.close()
	labeled_train_x = train_x[idx, :]
	labeled_train_y = train_y[idx, :]
	labeled_num_train = labeled_train_x.shape[0]

	with open('..\\data\\usvtrain.p', 'rb') as fp:
		idx = pickle.load(fp)
		fp.close()
	unlabeled_train_x = train_x[idx, :]
	unlabeled_train_y = train_y[idx, :]
	unlabeled_num_train = unlabeled_train_x.shape[0]

	labeled_train_data = tf.data.Dataset.from_tensor_slices((labeled_train_x, labeled_train_y))
	labeled_train_data = labeled_train_data.shuffle(labeled_num_train)
	labeled_train_data = labeled_train_data.map(train_parse, num_parallel_calls=8)
	labeled_train_data = labeled_train_data.batch(batch_size)
	labeled_train_iter = labeled_train_data.make_initializable_iterator()
	x_labeled, y_labeled = labeled_train_iter.get_next()

	unlabeled_train_data = tf.data.Dataset.from_tensor_slices((unlabeled_train_x, unlabeled_train_y))
	unlabeled_train_data = unlabeled_train_data.shuffle(unlabeled_num_train)
	unlabeled_train_data = unlabeled_train_data.map(train_parse, num_parallel_calls=8)
	unlabeled_train_data = unlabeled_train_data.batch(batch_size)
	unlabeled_train_iter = unlabeled_train_data.make_initializable_iterator()
	x_unlabeled, y_unlabeled = unlabeled_train_iter.get_next()

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

	with tf.name_scope("Classification_Loss"):
		clf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_std, labels=Y))
	sum_clfloss_op = tf.summary.scalar("Classification_Loss", clf_loss)

	with tf.name_scope("Consistency_Loss"):
		cst_loss = tf.losses.mean_squared_error(pred_std, pred_tc)
	sum_cstloss_op = tf.summary.scalar("Consistency_Loss", cst_loss)

	loss_op = tf.cond(is_labeled, lambda: tf.add(clf_loss, cst_loss), lambda: cst_loss)

	weights_std = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Student")
	weights_tc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Teacher")
	weights = zip(weights_tc, weights_std)

	# Learning Rate
	with tf.variable_scope('learning_rate'):
		lr_v = tf.Variable(lr_init, trainable=False)

	# Optimizer
	optimizer = tf.train.AdamOptimizer(lr_v)
	gvs = optimizer.compute_gradients(loss=loss_op, var_list=weights_std)
	capped_gvs = [(tf.clip_by_value(grad,-1.0, 1.0), var) for grad, var in gvs]
	opt_op = optimizer.apply_gradients(capped_gvs)

	# Create an Exponential Moving Average object
	ema = tf.train.ExponentialMovingAverage(decay=0.999)
	with tf.control_dependencies([opt_op]):
		train_op = ema.apply(weights_std)

	ema_op = tf.group([tf.assign(w_tc, ema.average(w_std)) for w_tc, w_std in weights])

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
		log = "\n========== Semi-supervised Training Begin ==========\n"
		write_logs(logs_usv, log, True)
		semi_train_start = time.time()

		# Initialize variables
		sess.run(tf.global_variables_initializer())

		# Op to write logs to Tensorboard
		train_sum_writer = tf.summary.FileWriter(logs_dir, tf.get_default_graph())
		
		log = "\n========== Labeled Training Begin ==========\n"
		write_logs(logs_usv, log, False)
		labeled_start = time.time()
		labeled_num_batches = int(math.ceil(labeled_num_train/batch_size))
		for epoch in range(labeled_num_epochs):
			epoch_start = time.time()
			avg_clf_loss = 0
			avg_cst_loss = 0
			avg_acc_std = 0
			avg_acc_tc = 0
			sess.run(labeled_train_iter.initializer)
			for batch in range(labeled_num_batches):
				batch_start = time.time()
				bx, by = sess.run([x_labeled, y_labeled])
				sess.run(train_op, feed_dict={X:bx, Y:by, drop_rate:dropout, is_labeled:True})
				sess.run(ema_op)
				clf_loss_val, cst_loss_val, sum_clfloss, sum_cstloss, acc_std_val, acc_tc_val, sum_acc =\
					sess.run([clf_loss, cst_loss, sum_clfloss_op, sum_cstloss_op, acc_std, acc_tc, sum_acc_op], feed_dict={X:bx, Y:by, drop_rate:0.0})
				avg_clf_loss += clf_loss_val
				avg_cst_loss += cst_loss_val
				avg_acc_std += acc_std_val
				avg_acc_tc += acc_tc_val
				train_sum_writer.add_summary(sum_clfloss, epoch*labeled_num_batches+batch)
				train_sum_writer.add_summary(sum_cstloss, epoch*labeled_num_batches+batch)
				train_sum_writer.add_summary(sum_acc, epoch*labeled_num_batches+batch)
				log = "Time {:2.5f}, Epoch {}, Batch {}, Classification Loss = {:2.5f}, Consistency Loss = {:1.9f}, Student Accuracy = {:0.4f}, Teacher Accuracy = {:0.4f}"\
					.format(time.time()-batch_start, epoch, batch, clf_loss_val, cst_loss_val, acc_std_val, acc_tc_val)
				write_logs(logs_usv, log, False)
			log = "\nTime {:2.5f}, Epoch {}, Average Classification Loss = {:2.5f}, Average Consistency Loss = {:1.9f}, Average Student Accuracy = {:0.4f}, Average Teacher Accuracy = {:0.4f}\n"\
				.format(time.time()-epoch_start, epoch, avg_clf_loss/labeled_num_batches, avg_cst_loss/labeled_num_batches, avg_acc_std/labeled_num_batches, avg_acc_tc/labeled_num_batches)
			write_logs(logs_usv, log, False)
		log = "\nLabeled Training Time: {:2.5f}".format(time.time()-labeled_start)
		write_logs(logs_usv, log, False)
		log = "\n========== Labeled Training End ==========\n"
		write_logs(logs_usv, log, False)
		
		log = "\n========== Unlabeled Training Begin ==========\n"
		write_logs(logs_usv, log, False)
		unlabeled_start = time.time()
		unlabeled_num_batches = int(math.ceil(unlabeled_num_train/batch_size))
		for epoch in range(labeled_num_epochs, labeled_num_epochs+unlabeled_num_epoches):
			epoch_start = time.time()
			
			if (epoch > lr_start_decay) and (epoch % lr_decay_period == 0):
				new_lr = lr_v * lr_decay
				sess.run(tf.assign(lr_v, new_lr))
				log = "** New learning rate: %1.9f **\n" % (lr_v.eval())
				write_logs(logs_usv, log, False)
			elif epoch == 0:
				sess.run(tf.assign(lr_v, lr_init))
				log = "** Initial learning rate: %1.9f **\n" % (lr_init)
				write_logs(logs_usv, log, False)
			
			avg_cst_loss = 0
			avg_acc_std = 0
			avg_acc_tc = 0

			sess.run(unlabeled_train_iter.initializer)
			for batch in range(unlabeled_num_batches):
				batch_start = time.time()

				bx, by = sess.run([x_unlabeled, y_unlabeled])
				sess.run(train_op, feed_dict={X:bx, Y:by, drop_rate:dropout, is_labeled:False})
				sess.run(ema_op)
				cst_loss_val, sum_cstloss, acc_std_val, acc_tc_val, sum_acc = sess.run([cst_loss, sum_cstloss_op, acc_std, acc_tc, sum_acc_op], feed_dict={X:bx, Y:by, drop_rate:0.0})

				avg_cst_loss += cst_loss_val
				avg_acc_std += acc_std_val
				avg_acc_tc += acc_tc_val

				train_sum_writer.add_summary(sum_cstloss, epoch*unlabeled_num_batches+batch)
				train_sum_writer.add_summary(sum_acc, epoch*unlabeled_num_batches+batch)

				log = "Time {:2.5f}, Epoch {}, Batch {}, Consistency Loss = {:1.9f}, Student Accuracy = {:0.4f}, Teacher Accuracy = {:0.4f}"\
					.format(time.time()-batch_start, epoch, batch, cst_loss_val, acc_std_val, acc_tc_val)
				write_logs(logs_usv, log, False)

			log = "\nTime {:2.5f}, Epoch {}, Average Consistency Loss = {:2.5f}, Average Student Accuracy = {:0.4f}, Average Teacher Accuracy = {:0.4f}\n"\
				.format(time.time()-epoch_start, epoch, avg_cst_loss/unlabeled_num_batches, avg_acc_std/unlabeled_num_batches, avg_acc_tc/unlabeled_num_batches)
			write_logs(logs_usv, log, False)
		log = "\nUnlabeled Training Time: {:2.5f}".format(time.time()-unlabeled_start)
		write_logs(logs_usv, log, False)
		log = "\nSemi-supervised Training Time: {:2.5f}".format(time.time()-semi_train_start)
		write_logs(logs_usv, log, False)
		log = "\n========== Semi-supervised Training End ==========\n"
		write_logs(logs_usv, log, False)

		# Save model
		save_path = saver.save(sess, usv_model_dir)
		log = "Model is saved in file: %s" % save_path
		write_logs(logs_usv, log, False)

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


