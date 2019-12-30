import pickle
import numpy as np
import tensorflow as tf

from config import config

# CIFAR-10 Parameters
height = config.IMG.height
width = config.IMG.width

def write_logs(filename, log, start=False):
  print(log)
  if start == True:
    f = open(filename, 'w')
    f.write(log + '\n')
  else:
    f = open(filename, 'a')
    f.write(log + '\n')
    f.close()

def dense_to_one_hot(labels_dense, num_classes=10):
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot

def get_data_set(name="train"):
	x = None
	y = None

	if name is "train":
		for i in range(5):
			f = open('..\\data\\cifar-10-batches-py\\data_batch_'+str(i+1), 'rb')
			datadict = pickle.load(f, encoding='latin1')
			f.close()

			x_ = datadict["data"]
			y_ = datadict["labels"]

			x_ = np.array(x_, dtype=float) / 255.0
			x_ = x_.reshape([-1, 3, 32, 32])
			x_ = x_.transpose([0, 2, 3, 1])
			x_ = x_.reshape(-1, 32*32*3)

			if x is None:
				x = x_
				y = y_
			else:
				x = np.concatenate((x, x_), axis=0)
				y = np.concatenate((y, y_), axis=0)
	elif name is "test":
		f = open('..\\data\\cifar-10-batches-py\\test_batch', 'rb')
		datadict = pickle.load(f, encoding='latin1')
		f.close()

		x = datadict["data"]
		y = np.array(datadict["labels"])

		x = np.array(x, dtype=float) / 255.0
		x = x.reshape([-1, 3, 32, 32])
		x = x.transpose([0, 2, 3, 1])
		x = x.reshape([-1, 32*32*3])

	return x, dense_to_one_hot(y)

def test_parse(x, y):
	x = tf.reshape(x, [height, width, 3])
	x = tf.cast(x, tf.float32)
	return x, y

def train_parse(x, y):
	x = tf.reshape(x, [height, width, 3])
	x = tf.image.random_flip_left_right(x)
	x = tf.cast(x, tf.float32)
	noise = tf.random_normal(tf.shape(x), mean=0.0, stddev=0.15, dtype=tf.float32)
	x = tf.add(x, noise)
	x = tf.clip_by_value(x, 0.0, 255.0)
	y = tf.cast(y, tf.int32)
	return x, y
